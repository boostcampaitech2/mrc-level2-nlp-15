import pandas as pd
import torch
import torch.nn.functional as F
from transformers import (
    BertModel,
    AutoConfig,
    BertPreTrainedModel,
    AdamW,
    AutoTokenizer,
    set_seed,
    TrainingArguments,
    get_linear_schedule_with_warmup,
    AutoModel,
)
from datasets import load_metric, load_from_disk, Dataset, DatasetDict
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from models import BertEncoder
from tqdm import tqdm
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import random
import os
import re

"""
1 - dense training
2- dense embedding
3- faiss clustering
"""


def proprecessing(text):
    new_text = text.replace(r"\n\n", "")
    return new_text


class gold_train:
    def __init__(self, args):
        self.args = args
        dataset = load_from_disk(args.dataset_name)
        train_dataset = dataset["train"]
        valid_dataset = dataset["validation"]

        self.train_dataset = {}
        self.train_dataset["context"] = [
            proprecessing(string) for string in train_dataset["context"]
        ]
        self.train_dataset["question"] = train_dataset["question"]
        self.valid_dataset = {}
        self.valid_dataset["context"] = [
            proprecessing(string) for string in valid_dataset["context"]
        ]
        self.valid_dataset["question"] = valid_dataset["question"]

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        self.p_encoder = BertEncoder.from_pretrained(args.model_name)
        self.q_encoder = BertEncoder.from_pretrained(args.model_name)
        self.p_encoder.to(args.device)
        self.q_encoder.to(args.device)
        self.indexer = None  # build_faiss()로 생성합니다.

    def train(self, epochs, learning_rate, train_dataset=None):
        if not train_dataset:
            train_dataset = self.train_dataset
        # train dataset, train dataloader
        p_seqs = self.tokenizer(
            train_dataset["context"], padding="max_length", truncation=True, return_tensors="pt"
        )
        q_seqs = self.tokenizer(
            train_dataset["question"],
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # print(q_seqs[0])
        train_dataset = TensorDataset(
            p_seqs["input_ids"],
            p_seqs["attention_mask"],
            p_seqs["token_type_ids"],
            q_seqs["input_ids"],
            q_seqs["attention_mask"],
            q_seqs["token_type_ids"],
        )
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size)
        # valid dataset, valid dataloader
        q_seqs = self.tokenizer(
            self.valid_dataset["question"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        p_seqs = self.tokenizer(
            self.valid_dataset["context"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # print(q_seqs[0])
        valid_dataset = TensorDataset(
            p_seqs["input_ids"],
            p_seqs["attention_mask"],
            p_seqs["token_type_ids"],
            q_seqs["input_ids"],
            q_seqs["attention_mask"],
            q_seqs["token_type_ids"],
        )
        valid_loader = DataLoader(valid_dataset, batch_size=self.args.batch_size)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.p_encoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.p_encoder.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in self.q_encoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.q_encoder.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            # eps=args.adam_epsilon
        )
        # t_total = len(train_loader) // args.gradient_accumulation_steps * args.num_train_epochs
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=args.warmup_steps,
        #     num_training_steps=t_total
        # )
        best = 0
        for epoch in range(epochs):
            step = 0
            losses = 0
            for idx, data in enumerate(tqdm(train_loader)):
                step += 1
                p_inputs = {
                    "input_ids": data[0].to(self.args.device),
                    "attention_mask": data[1].to(self.args.device),
                    "token_type_ids": data[2].to(self.args.device),
                }
                q_inputs = {
                    "input_ids": data[3].to(self.args.device),
                    "attention_mask": data[4].to(self.args.device),
                    "token_type_ids": data[5].to(self.args.device),
                }
                targets = torch.arange(0, len(p_inputs["input_ids"])).long().to(self.args.device)
                q_output = self.q_encoder(**q_inputs)
                p_output = self.p_encoder(**p_inputs)
                retrieval = torch.matmul(q_output, p_output.T)
                retrieval_scores = F.log_softmax(retrieval, dim=1)

                loss = F.nll_loss(retrieval_scores, targets)
                losses += loss.item()
                # q_encoder.zero_grad()
                self.p_encoder.zero_grad()
                self.q_encoder.zero_grad()

                loss.backward()
                optimizer.step()
                # scheduler.step()
                if step % 100 == 0:
                    print(f"{epoch}epoch loss: {losses/(step)}")

                    losses = 0
                    correct = 0
                    step = 0
            optimizer.lr = 5e-6
            valid_embedding = self.dense_embedding(self.valid_dataset)
            valid_ans = []
            for idx, data in enumerate(tqdm(valid_loader)):
                with torch.no_grad():
                    q_inputs = {
                        "input_ids": data[3].to(self.args.device),
                        "attention_mask": data[4].to(self.args.device),
                        "token_type_ids": data[5].to(self.args.device),
                    }
                    output = self.p_encoder(**q_inputs)
                    if idx == 0:
                        query_vec = output
                    else:
                        query_vec = torch.cat((query_vec, output), 0)

            result = torch.mm(query_vec, valid_embedding.T)
            result = result.cpu().detach().numpy()
            doc_indices = []
            for i in range(result.shape[0]):
                sorted_result = np.argsort(result[i, :])[::-1]
                doc_indices.append(sorted_result.tolist()[:3])
            correct = 0
            for idx, i in enumerate(doc_indices):
                if idx in i:
                    correct += 1
            print(f"acc :{correct/len(valid_dataset)}")
        print("save")
        torch.save(self.p_encoder.state_dict(), "gold/p_encoder.pt")
        torch.save(self.q_encoder.state_dict(), "gold/q_encoder.pt")
        self.p_embedding = self.dense_embedding(self.train_dataset)

    def dense_embedding(self, dataset):
        """문맥의 임베딩 값을 구하고리턴합니다."""
        q_seqs = self.tokenizer(
            dataset["question"], padding="max_length", truncation=True, return_tensors="pt"
        )
        p_seqs = self.tokenizer(
            dataset["context"], padding="max_length", truncation=True, return_tensors="pt"
        )
        # print(q_seqs[0])
        dataset = TensorDataset(
            p_seqs["input_ids"],
            p_seqs["attention_mask"],
            p_seqs["token_type_ids"],
            q_seqs["input_ids"],
            q_seqs["attention_mask"],
            q_seqs["token_type_ids"],
        )
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size)
        print("Build passage embedding")
        for idx, data in enumerate(tqdm(dataloader)):
            p_inputs = {
                "input_ids": data[0].to("cuda"),
                "attention_mask": data[1].to("cuda"),
                "token_type_ids": data[2].to("cuda"),
            }
            with torch.no_grad():
                p_inputs = {k: v for k, v in p_inputs.items()}
                output = self.p_encoder(**p_inputs)
                if idx == 0:
                    p_embedding = output
                else:
                    p_embedding = torch.cat((p_embedding, output), 0)
        return p_embedding

    def faiss_clustering(self, k=2, num_clusters=64):
        """return faise로 self.train_datset에서 유사도 k개로 뽑은 문맥들을 추가한 dataset
        return example:
         (dataset[0],datset[0]과 유사한 trainset내의 문서,...)
         총 k * len(dataset)의 크기의 dataset
        """

        assert self.p_embedding is not None, "embedding을 먼저 진행해야해요."

        p_emb = self.p_embedding.cpu().detach().numpy()
        emb_dim = p_emb.shape[-1]

        quantizer = faiss.IndexFlatL2(emb_dim)

        self.indexer = faiss.IndexIVFScalarQuantizer(
            quantizer, quantizer.d, num_clusters, faiss.METRIC_L2
        )
        self.indexer.train(p_emb)
        self.indexer.add(p_emb)
        D, I = self.indexer.search(p_emb, k)
        train_dataset = {"context": [], "question": []}
        for n, i in enumerate(I):
            context = [self.train_dataset["context"][idx] for idx in i]
            question = [self.train_dataset["question"][idx] for idx in i]
            if n not in i:
                context[0] = self.train_dataset["context"][n]
                question[0] = self.train_dataset["question"][n]
            train_dataset["context"].extend(context)
            train_dataset["question"].extend(question)
        return train_dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--dataset_name", type=str, default="../data/train_dataset", help="dataset")
    parser.add_argument("--model_name", type=str, default="klue/bert-base", help="model name")
    parser.add_argument("--batch_size", type=int, default=22, help="model name")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--epochs", type=int, default=1, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=10e-5, help="dataset")
    parser.add_argument("--warmup_steps", type=int, default=500, help="dataset")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="dataset")

    args = parser.parse_args()

    x = gold_train(args)
    x.train(1, 5e-5)

    dataset = x.faiss_clustering()
    x.train(3, 10e-5, dataset)
