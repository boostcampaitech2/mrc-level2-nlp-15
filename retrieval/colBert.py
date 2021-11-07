# import python modules
import json
from typing import List
import string
import pickle
import os
import os.path as p
import datetime
import itertools

# import data wrangling modules
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import random

# import torch modules
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# import transformers and its related modules
from transformers import (
    AutoTokenizer,
    BertModel,
    RobertaModel,
    BertPreTrainedModel,
    AdamW,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    AdamW,
    TrainingArguments,
    BertPreTrainedModel,
    BertModel,
    BertTokenizerFast,
    BertConfig,
)
from datasets import (
    Dataset,
    load_from_disk,
)

# import third party modules
from collections import OrderedDict, defaultdict

# import custom modules
from utils.utils_qa import CustomSampler


def seed():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)


# https://github.com/stanford-futuredata/ColBERT/tree/master/colbert
class QueryTokenizer:
    def __init__(self, model_checkpoint):
        self.tok = BertTokenizerFast.from_pretrained(model_checkpoint)

        # "[Q]" = query, doc 구분을 위한 special token
        self.Q_marker_token, self.Q_marker_token_id = (
            "[Q]",
            self.tok.convert_tokens_to_ids("[unused0]"),
        )
        self.cls_token, self.cls_token_id = self.tok.cls_token, self.tok.cls_token_id
        self.sep_token, self.sep_token_id = self.tok.sep_token, self.tok.sep_token_id
        self.mask_token, self.mask_token_id = (
            self.tok.mask_token,
            self.tok.mask_token_id,
        )
        self.query_maxlen = self.tok.model_max_length

    def tokenize(self, batch_text, add_special_tokens=False):

        assert type(batch_text) in [list, tuple], type(batch_text)

        tokens = [self.tok.tokenize(x, add_special_tokens=False) for x in batch_text]

        if not add_special_tokens:
            return tokens

        prefix, suffix = [self.cls_token, self.Q_marker_token], [self.sep_token]

        tokens = [
            prefix  # [Cls]+[Q]
            + lst
            + suffix
            + [self.mask_token]
            * (
                self.query_maxlen - (len(lst) + 3)
            )  # query augmentation: padding으로 [mask] 토큰 추가
            for lst in tokens
        ]

        return tokens

    def encode(self, batch_text, add_special_tokens=False):
        """
        return input_ids(query)
        """
        assert type(batch_text) in [list, tuple], type(batch_text)

        ids = self.tok(batch_text, add_special_tokens=False)["input_ids"]

        if not add_special_tokens:
            return ids

        prefix, suffix = (
            [self.cls_token_id, self.Q_marker_token_id],
            [self.sep_token_id],
        )
        ids = [
            prefix
            + lst
            + suffix
            + [self.mask_token_id] * (self.query_maxlen - (len(lst) + 3))
            for lst in ids
        ]

        return ids

    def tensorize(self, batch_text, bsize=None):
        """
        return input_ids, attention_mask(query)
        """
        assert type(batch_text) in [list, tuple], type(batch_text)

        # add placehold for the [Q] marker
        batch_text = [". " + x for x in batch_text]

        obj = self.tok(
            batch_text,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            max_length=self.tok.model_max_length,
        )

        ids, mask = obj["input_ids"], obj["attention_mask"]

        # postprocess for the [Q] marker and the [MASK] augmentation
        ids[:, 1] = self.Q_marker_token_id
        ids[ids == 0] = self.mask_token_id

        if bsize:
            batches = _split_into_batches(ids, mask, bsize)
            return batches

        return ids, mask


class DocTokenizer:
    def __init__(self, model_checkpoint):
        self.tok = BertTokenizerFast.from_pretrained(model_checkpoint)

        # "[D]" = query, doc 구분을 위한 special token
        self.D_marker_token, self.D_marker_token_id = (
            "[D]",
            self.tok.convert_tokens_to_ids("[unused1]"),
        )
        self.cls_token, self.cls_token_id = self.tok.cls_token, self.tok.cls_token_id
        self.sep_token, self.sep_token_id = self.tok.sep_token, self.tok.sep_token_id

        # assert self.D_marker_token_id == 1

    def tokenize(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], type(batch_text)

        tokens = [self.tok.tokenize(x, add_special_tokens=False) for x in batch_text]

        if not add_special_tokens:
            return tokens

        prefix, suffix = [self.cls_token, self.D_marker_token], [self.sep_token]
        tokens = [prefix + lst + suffix for lst in tokens]

        return tokens

    def encode(self, batch_text, add_special_tokens=False):
        """
        return input_ids(doc)
        """
        assert type(batch_text) in [list, tuple], type(batch_text)

        ids = self.tok(batch_text, add_special_tokens=False)["input_ids"]

        if not add_special_tokens:
            return ids

        prefix, suffix = (
            [self.cls_token_id, self.D_marker_token_id],
            [self.sep_token_id],
        )
        ids = [prefix + lst + suffix for lst in ids]

        return ids

    def tensorize(self, batch_text, bsize=None):
        """
        return input_ids, attention_mask(doc)
        """
        assert type(batch_text) in [list, tuple], type(batch_text)

        # add placehold for the [D] marker
        batch_text = [". " + x for x in batch_text]

        obj = self.tok(
            batch_text,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=self.tok.model_max_length,
        )

        ids, mask = obj["input_ids"], obj["attention_mask"]

        # postprocess for the [D] marker
        ids[:, 1] = self.D_marker_token_id

        if bsize:
            # length based bucketing
            ids, mask, reverse_indices = _sort_by_length(ids, mask, bsize)
            batches = _split_into_batches(ids, mask, bsize)
            return batches, reverse_indices

        return ids, mask


def tensorize_triples(
    query_tokenizer, doc_tokenizer, queries, positives, negatives, bsize
):
    """
    return dataloader
    """
    # assert len(queries) == len(positives) == len(negatives)
    # assert bsize is None or len(queries) % bsize == 0

    N = len(queries)
    queries = queries.to_list()
    Q_ids, Q_mask = query_tokenizer.tensorize(queries)

    positives = positives.to_list()
    negatives = negatives.to_list()

    D_ids, D_mask = doc_tokenizer.tensorize(positives + negatives)
    D_ids, D_mask = D_ids.view(2, N, -1), D_mask.view(2, N, -1)

    # Compute max among {length of i^th positive, length of i^th negative} for i \in N
    maxlens = D_mask.sum(-1).max(0).values

    # Sort by maxlens
    indices = maxlens.sort().indices
    Q_ids, Q_mask = Q_ids[indices], Q_mask[indices]
    D_ids, D_mask = D_ids[:, indices], D_mask[:, indices]

    (positive_ids, negative_ids), (positive_mask, negative_mask) = D_ids, D_mask

    query_batches = _split_into_batches(Q_ids, Q_mask, bsize)
    positive_batches = _split_into_batches(positive_ids, positive_mask, bsize)
    negative_batches = _split_into_batches(negative_ids, negative_mask, bsize)

    batches = []
    for (q_ids, q_mask), (p_ids, p_mask), (n_ids, n_mask) in zip(
        query_batches, positive_batches, negative_batches
    ):
        Q = (torch.cat((q_ids, q_ids)), torch.cat((q_mask, q_mask)))
        D = (torch.cat((p_ids, n_ids)), torch.cat((p_mask, n_mask)))
        batches.append((Q, D))

    return batches


def _sort_by_length(ids, mask, bsize):
    """
    비슷한 길이를 batch에 주기 위한 length based bucketing. 
    """
    if ids.size(0) <= bsize:
        return ids, mask, torch.arange(ids.size(0))

    indices = mask.sum(-1).sort().indices
    reverse_indices = indices.sort().indices

    return ids[indices], mask[indices], reverse_indices


def _split_into_batches(ids, mask, bsize):
    batches = []
    for offset in range(0, ids.size(0), bsize):
        batches.append((ids[offset : offset + bsize], mask[offset : offset + bsize]))

    return batches


def timestamp():
    format_str = "%Y-%m-%d_%H.%M.%S"
    result = datetime.datetime.now().strftime(format_str)
    return result


def batch(group, bsize, provide_offset=False):
    offset = 0
    while offset < len(group):
        L = group[offset : offset + bsize]
        yield ((offset, L) if provide_offset else L)
        offset += len(L)
    return


class ColBERT(BertPreTrainedModel):
    def __init__(
        self,
        config,
        mask_punctuation=string.punctuation,
        dim=128,
        similarity_metric="cosine",
        model_checkpoint="klue/bert-base",
    ):
        super(ColBERT, self).__init__(config)

        self.similarity_metric = similarity_metric
        self.dim = dim

        self.mask_punctuation = mask_punctuation
        self.skiplist = {}

        if self.mask_punctuation:
            self.tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint)
            self.skiplist = {
                w: True
                for symbol in mask_punctuation
                for w in [
                    symbol,
                    self.tokenizer.encode(symbol, add_special_tokens=False)[0],
                ]
            }

        self.bert = BertModel(config)
        self.linear = nn.Linear(config.hidden_size, dim, bias=False)

        self.init_weights()

    def forward(self, Q=None, D=None):
        # return self.query(**Q), self.doc(**D)
        return self.score(self.query(**Q), self.doc(**D))

    def query(self, input_ids, attention_mask=None):
        """
        query encoder = Normalize(CNN(BERT)
        """
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)
        Q = torch.nn.functional.normalize(Q, p=2, dim=2)
        return Q

    def doc(self, input_ids, attention_mask=None):
        """
        doc encoder = Filter(Normalize(CNN(BERT)))
        """
        D = self.bert(input_ids, attention_mask=attention_mask)[0]
        D = self.linear(D)

        # Filter (문장부호 제거)
        mask = torch.tensor(self.mask(input_ids), device="cuda").unsqueeze(2).float()
        D = D * mask
        D = torch.nn.functional.normalize(D, p=2, dim=2)

        return D

    def score(self, Q, D):
        """
        relevance score (cosine similarity / l2 distance)
        """
        if self.similarity_metric == "cosine":
            return (Q @ D.permute(0, 2, 1)).max(2).values.sum(1)

        assert self.similarity_metric == "l2"
        return (
            (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1)) ** 2).sum(-1))
            .max(-1)
            .values.sum(-1)
        )

    def mask(self, input_ids):
        mask = [
            [(x not in self.skiplist) and (x != 0) for x in d]
            for d in input_ids.cpu().tolist()
        ]
        return mask


class ColBertRetrieval:
    def __init__(self, args, dataset, q_tokenizer, d_tokenizer, colbert_encoder, df):

        self.args = args
        self.dataset = dataset

        self.q_tokenizer = q_tokenizer
        self.d_tokenizer = d_tokenizer
        self.colbert_encoder = colbert_encoder

        self.df = df
        self.colbert_p_embedding = None

    def get_p_embs(self, corpus=None, colbert_encoder=None):
        """
        colbert encoder의 document embedding을 save/load하는 함수
        """

        # save pickle
        pickle_name = f"colbert_p_embedding.bin"
        emd_path = os.path.join(self.data_path, pickle_name)

        if os.path.isfile(emd_path) and os.path.isfile(emd_path):
            with open(emd_path, "rb") as file:
                self.colbert_p_embedding = pickle.load(file)
        else:
            if colbert_encoder is None:
                colbert_encoder = self.colbert_encoder
            if corpus is None:
                with open(
                    "../data/wikipedia_documents.json", "r", encoding="utf-8"
                ) as f:
                    wiki = json.load(f)

                corpus = list(dict.fromkeys([v["text"] for v in wiki.values()]))

            with torch.no_grad():
                colbert_encoder.eval()

                p_embs = []
                for p in tqdm(corpus):
                    # p = tokenizer(p, padding='max_length', truncation=True, return_tensors='pt').to('cuda')
                    # p_emb = p_encoder(**p).to('cpu').numpy()
                    p_emb = colbert_encoder.doc(
                        self.d_tokenizer.tensorize([p])[0].to("cuda"),
                        self.d_tokenizer.tensorize([p])[1].to("cuda"),
                    )
                    p_emb = p_emb.to("cpu").numpy()
                    p_embs.append(p_emb)
            p_embs = torch.Tensor(p_embs).squeeze()

            self.colbert_p_embedding = p_embs
            with open(emd_path, "wb") as file:
                pickle.dump(self.colbert_p_embedding, file)

        return self.colbert_p_embedding

    def get_relavant_doc(self, queries, colbert_encoder, k=1):
        """
        query의 top=k relevant의 indices, score를 리턴하는 함수
        """

        self.get_p_embs(self, colbert_encoder=colbert_encoder)

        with torch.no_grad():
            colbert_encoder.eval()
            q_embs = []
            for q in tqdm(queries):
                q_emb = colbert_encoder.doc(
                    self.q_tokenizer.tensorize([q])[0].to("cuda"),
                    self.q_tokenizer.tensorize([q])[1].to("cuda"),
                )
                q_emb = q_emb.to("cpu")
                q_embs.append(q_emb)

        result_scores = []
        result_indices = []
        for i, qq in enumerate(tqdm(q_embs)):
            dot_prod_scores = colbert_encoder.score(qq, self.colbert_p_embedding)
            score, indice = torch.sort(torch.tensor(dot_prod_scores), descending=True)
            result_scores.append(score)
            result_indices.append(indice)

        return result_scores, result_indices

    def train(self, args=None, tokenizer=None, df=None):
        if args is None:
            args = self.args
        if tokenizer is None:
            tokenizer = self.tokenizer

        # [query, p+, p-]
        train_dataloader = tensorize_triples(
            self.q_tokenizer,
            self.d_tokenizer,
            self.df["question"],
            self.df["original_context"],
            self.df["context"],
            self.args.per_device_train_batch_size,
        )

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.colbert_encoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.colbert_encoder.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            # eps=args.adam_epsilon
        )

        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

        global_step = 0

        self.colbert_encoder.zero_grad()

        train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
        self.colbert_encoder.train()

        criterion = nn.CrossEntropyLoss()

        best_acc = 0
        for epoch, _ in enumerate(train_iterator):
            losses = 0
            for step, batch in enumerate(train_dataloader):

                # D
                p_inputs = {
                    "input_ids": batch[0][0].cuda(),
                    "attention_mask": batch[0][1].cuda(),
                }

                # Q
                q_inputs = {
                    "input_ids": batch[1][0].cuda(),
                    "attention_mask": batch[1][1].cuda(),
                }

                sim_scores = self.colbert_encoder(Q=q_inputs, D=p_inputs)

                # Calculate similarity score & loss
                targets = torch.zeros(args.per_device_train_batch_size).long()

                if torch.cuda.is_available():
                    targets = targets.to("cuda")

                sim_scores = sim_scores.view(-1, 2)

                # get mean of the loss
                loss = criterion(sim_scores, targets)
                losses += loss.item()
                if step % 100 == 0:
                    print(
                        f"{epoch}epoch loss: {losses/(step+1)}"
                    )  # Accumulation할 경우 주석처리

                self.colbert_encoder.zero_grad()

                loss.backward()
                optimizer.step()
                scheduler.step()

                global_step += 1

                # torch.cuda.empty_cache()
                del p_inputs, q_inputs

            acc = self.evaluate(args=args, colbert_encoder=self.colbert_encoder)
            if acc > best_acc:
                torch.save(self.colbert_encoder, "./colbert_encoder.pt")
                best_acc = acc

        return self.colbert_encoder

    def evaluate(self, args=None, colbert_encoder=None, corpus=None, topk=5):

        if colbert_encoder is None:
            colbert_encoder = self.colbert_encoder
        if corpus is None:
            with open("../data/wikipedia_documents.json", "r", encoding="utf-8") as f:
                wiki = json.load(f)

            corpus = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        doc_scores, doc_indices = self.get_relavant_doc(
            args.dataset["validation"]["question"], colbert_encoder
        )

        total = []
        for idx, example in enumerate(
            tqdm(args.dataset["validation"], desc="Dense retrieval: ")
        ):
            tmp = {
                # Query와 해당 id를 반환합니다.
                "question": example["question"],
                "id": example["id"],
                # Retrieve한 Passage의 id, context를 반환합니다.
                "context_id": doc_indices[idx][:topk],
                "context": " ".join(  # 기존에는 ' '.join()
                    [corpus[pid] for pid in doc_indices[idx][:topk]]
                ),
                "scores": doc_scores[idx][:topk],
            }
            if "context" in example.keys() and "answers" in example.keys():
                # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                tmp["original_context"] = example["context"]
                tmp["answers"] = example["answers"]
            total.append(tmp)

        cqas = pd.DataFrame(total)

        correct_cnt = 0
        for i in range(len(cqas)):
            if cqas["original_context"][i] in cqas["context"][i]:
                correct_cnt += 1

        return correct_cnt / len(self.dataset["validation"])


def main(dataset_path):
    args = TrainingArguments(
        output_dir="dense_retireval",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=3,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        num_train_epochs=10,
        weight_decay=0.01,
    )
    model_checkpoint = "klue/bert-base"
    dataset = load_from_disk(dataset_path)
    train_dataset = dataset["train"]

    # load pre-trained model on cuda (if available)
    colbert_encoder = ColBERT.from_pretrained(model_checkpoint).to(args.device)

    # load tokenizer
    q_tokenizer = QueryTokenizer(model_checkpoint)
    d_tokenizer = DocTokenizer(model_checkpoint)

    # load dataframe(es negative) !!!! add code
    df = pd.read_csv("../data/colbertdata_join_top10_wikipedia.csv")

    retriever = ColBertRetrieval(
        args=args,
        dataset=train_dataset,
        q_tokenizer=q_tokenizer,
        d_tokenizer=d_tokenizer,
        colbert_encoder=colbert_encoder,
        df=df,
    )

    colbert_encoder = retriever.train()


if __name__ == "__main__":
    dataset_path = '../dataset/train_dataset/"'
    seed()
    main(dataset_path)
