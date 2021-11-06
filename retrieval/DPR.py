# import python modules
import json

# import data wrangling modules
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import random

# import torch modules
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, SequentialSampler

# import transformers and its related modules
from transformers import (
    AutoTokenizer,
    BertModel,
    BertPreTrainedModel,
    AdamW,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)
from datasets import load_from_disk

# import third party modules
from collections import defaultdict

# import custom modules
from utils.utils_qa import CustomSampler
from elasticsearch import Elasticsearch, helpers


def seed():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)


class CustomDataset(Dataset):
    def __init__(self, queries, passages, wiki_passages, max_len=512):
        self.queries = queries
        self.passages = np.array(passages)
        self.tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
        self.max_seq_len = max_len
        self.question_max_len = 64
        self.wiki_passages = np.array(wiki_passages)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        p_with_neg = []
        quer = self.queries[idx]
        ground_truth = self.passages[idx]
        num_negs = 31
        while True:
            neg_idxs = np.random.randint(0, len(self.wiki_passages), size=num_negs)
            p_negs = self.wiki_passages[neg_idxs]
            if ground_truth not in p_negs:
                p_with_neg.append(ground_truth)
                p_with_neg.extend(p_negs)
                break
        quer_inputs = self.tokenizer(
            quer, padding="max_length", return_tensors="pt", truncation=True
        )
        pass_inputs = self.tokenizer(
            p_with_neg, padding="max_length", return_tensors="pt", truncation=True
        )

        return {
            "quer_inputs": quer_inputs,
            "passage_inputs": pass_inputs,
        }


class CustomDataset_Overflow(Dataset):
    """
    설명 적기
    """

    def __init__(self, queries, passages, wiki_passages, max_len=512):
        self.queries = queries
        self.passages = np.array(passages)
        self.tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
        self.max_seq_len = max_len
        self.question_max_len = 64
        self.wiki_passages = np.array(wiki_passages)

    def __len__(self):
        return len(self.queries)

    def _return_train_dataset(self):
        for i in range(len(self.queries)):
            if i == 0:
                q_seqs = self.tokenizer(
                    self.queries[i],
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                p_seqs = self.tokenizer(
                    self.passages[i],
                    truncation=True,
                    stride=128,
                    padding="max_length",
                    return_overflowing_tokens=True,
                    return_offsets_mapping=True,
                    return_tensors="pt",
                )
                p_seqs.pop("overflow_to_sample_mapping")
                p_seqs.pop("offset_mapping")

                for k in q_seqs.keys():
                    q_seqs[k] = q_seqs[k].tolist()
                    p_seqs[k] = p_seqs[k].tolist()
            else:
                tmp_q_seq = self.tokenizer(
                    self.queries[i],
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                tmp_p_seq = self.tokenizer(
                    self.passages[i],
                    truncation=True,
                    stride=128,
                    padding="max_length",
                    return_overflowing_tokens=True,
                    return_offsets_mapping=True,
                    return_tensors="pt",
                )

                tmp_p_seq.pop("overflow_to_sample_mapping")
                tmp_p_seq.pop("offset_mapping")

                for k in tmp_p_seq.keys():
                    tmp_p_seq[k] = tmp_p_seq[k].tolist()
                    tmp_q_seq[k] = tmp_q_seq[k].tolist()

                for j in range(len(tmp_p_seq["input_ids"])):
                    q_seqs["input_ids"].append(tmp_q_seq["input_ids"][0])
                    q_seqs["token_type_ids"].append(tmp_q_seq["token_type_ids"][0])
                    q_seqs["attention_mask"].append(tmp_q_seq["attention_mask"][0])
                    p_seqs["input_ids"].append(tmp_p_seq["input_ids"][j])
                    p_seqs["token_type_ids"].append(tmp_p_seq["token_type_ids"][j])
                    p_seqs["attention_mask"].append(tmp_p_seq["attention_mask"][j])

        for k in q_seqs.keys():
            q_seqs[k] = torch.tensor(q_seqs[k])
            p_seqs[k] = torch.tensor(p_seqs[k])

        train_dataset = TensorDataset(
            p_seqs["input_ids"],
            p_seqs["attention_mask"],
            p_seqs["token_type_ids"],
            q_seqs["input_ids"],
            q_seqs["attention_mask"],
            q_seqs["token_type_ids"],
        )
        return train_dataset


from torch.utils.data import DataLoader, TensorDataset


class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        pooled_output = outputs[1]

        return pooled_output


def evaluate(p_encoder, q_encoder, data_path, tokenizer):
    dataset = load_from_disk(data_path)
    p_embs = []
    search_corpus = list([example["context"] for example in dataset["validation"]])
    valid_p_seqs = tokenizer(
        search_corpus, padding="max_length", truncation=True, return_tensors="pt"
    )
    valid_dataset = TensorDataset(
        valid_p_seqs["input_ids"],
        valid_p_seqs["attention_mask"],
        valid_p_seqs["token_type_ids"],
    )
    valid_sampler = SequentialSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=1)

    with torch.no_grad():
        epoch_iterator = tqdm(
            valid_dataloader, desc="Iteration", position=0, leave=True
        )
        p_encoder.eval()

        for _, batch in enumerate(epoch_iterator):
            batch = tuple(t.cuda() for t in batch)

            p_inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            outputs = p_encoder(**p_inputs).to("cpu").numpy()
            p_embs.extend(outputs)

    p_embs = np.array(p_embs)
    query = dataset["validation"]["question"]
    ground_truth = dataset["validation"]["context"]
    valid_q_seqs = tokenizer(
        query, padding="max_length", truncation=True, return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        q_encoder.eval()
        q_embs = q_encoder(**valid_q_seqs).to("cpu").numpy()

    if torch.cuda.is_available():
        p_embs_cuda = torch.Tensor(p_embs).to("cuda")
        q_embs_cuda = torch.Tensor(q_embs).to("cuda")

    dot_prod_scores = torch.matmul(q_embs_cuda, torch.transpose(p_embs_cuda, 0, 1))
    rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
    k = 20
    score = 0

    for i, q in enumerate(query):
        r = rank[i]
        r_ = r[: k + 1]
        passages = [search_corpus[i] for i in r_]
        if ground_truth[i] in passages:
            score += 1
    accuracy = score / len(query)
    print(accuracy)
    return accuracy


def train(args, train_dataloader, p_model, q_model, model_checkpoint):

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in p_model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in p_model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
        {
            "params": [
                p
                for n, p in q_model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in q_model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    t_total = (
        len(train_dataloader)
        // args.gradient_accumulation_steps
        * args.num_train_epochs
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    p_model.zero_grad()
    q_model.zero_grad()
    torch.cuda.empty_cache()
    device = torch.device("cuda")
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    best_acc = 0
    for _ in train_iterator:
        epoch_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            a_inputs = {
                "input_ids": batch["quer_inputs"]["input_ids"].squeeze(1).to(device),
                "attention_mask": batch["quer_inputs"]["attention_mask"]
                .squeeze(1)
                .to(device),
                "token_type_ids": batch["quer_inputs"]["token_type_ids"]
                .squeeze(1)
                .to(device),
            }
            b_inputs = {
                "input_ids": batch["passage_inputs"]["input_ids"].squeeze(0).to(device),
                "attention_mask": batch["passage_inputs"]["attention_mask"]
                .squeeze(0)
                .to(device),
                "token_type_ids": batch["passage_inputs"]["token_type_ids"]
                .squeeze(0)
                .to(device),
            }

            pl1 = q_model(**a_inputs)
            pl2 = p_model(**b_inputs)
            targets = torch.tensor([0]).to(device)
            dot_prod_scores = torch.matmul(pl1, torch.transpose(pl2, 0, 1))
            sim_scores = dot_prod_scores.view(1, -1)
            sim_score2 = F.log_softmax(sim_scores, dim=1)
            loss = F.nll_loss(sim_score2, torch.tensor(targets))
            epoch_loss += loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            q_model.zero_grad()
            p_model.zero_grad()
            torch.cuda.empty_cache()
        print(epoch_loss / len(train_dataloader))
        acc = evaluate(p_model, q_model, dataset_path, tokenizer=tokenizer)
        if acc > best_acc:
            torch.save(p_model, "./p_model.pt")
            torch.save(q_model, "./q_model.pt")
            best_acc = acc
    return p_model, q_model


def main(dataset_path):
    args = TrainingArguments(
        output_dir="dense_retireval",
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        num_train_epochs=10,
        weight_decay=0.01,
    )
    model_checkpoint = "klue/bert-base"
    dataset = load_from_disk(dataset_path)

    context_path = "./data/wikipedia_documents.json"
    with open(context_path, "r", encoding="utf-8") as f:
        wiki = json.load(f)

    search_corpus = list(dict.fromkeys([v["text"] for v in wiki.values()]))

    passages = list([example["context"] for example in dataset["train"]])
    questions = list([example["question"] for example in dataset["train"]])

    if args.dpr_method == "large_negative":
        train_dataset = CustomDataset(questions, passages, search_corpus)
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    elif args.dpr_method == "overflow_with_no_truncation":
        custom_dataset = CustomDataset_Overflow(questions, passages, search_corpus)
        train_dataset = custom_dataset._return_train_dataset()
        sampler = CustomSampler(train_dataset, args.per_device_train_batch_size)
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.per_device_train_batch_size, sampler=sampler
        )

    # load pre-trained model on cuda (if available)
    p_encoder = BertEncoder.from_pretrained(model_checkpoint)
    q_encoder = BertEncoder.from_pretrained(model_checkpoint)

    if torch.cuda.is_available():
        p_encoder.cuda()
        q_encoder.cuda()

    train(args, train_dataloader, p_encoder, q_encoder, model_checkpoint)


if __name__ == "__main__":
    dataset_path = '../dataset/train_dataset/"'
    seed()
    main(dataset_path)
