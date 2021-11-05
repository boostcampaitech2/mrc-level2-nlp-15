from datasets import load_from_disk
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm, trange
import random
import torch
import torch.nn.functional as F
from transformers import (
    BertModel,
    BertPreTrainedModel,
    AdamW,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import SequentialSampler
import json
import yaml
from torch.utils.data import Dataset


def seed():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)


# Read config.yaml file
with open("config.yaml") as infile:
    SAVED_CFG = yaml.load(infile, Loader=yaml.FullLoader)
    SAVED_CFG = dotdict(SAVED_CFG)

# arguments setting
data_args = dotdict(SAVED_CFG.data)
model_args = dotdict(SAVED_CFG.custom_model)

# adding additional arguments
model_args.batch_size = 10
model_args.num_rnn_layers = 2
model_args.learning_rate = 2e-5
model_args.num_folds = 4
model_args.gamma = 1.0
model_args.smoothing = 0.2
model_args


class CustomDataset(Dataset):
    """make custom dataset for dense retrieval"""

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
    train_dataset = CustomDataset(questions, passages, search_corpus)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

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
