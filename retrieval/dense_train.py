import torch
import torch.nn.functional as F
from transformers import (
    BertModel,
    BertPreTrainedModel,
    AdamW,
    AutoTokenizer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)
from datasets import load_metric, load_from_disk, Dataset, DatasetDict
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from models import BertEncoder
from tqdm import tqdm
import random


def train(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    datasets = load_from_disk(args.dataset_name)
    train_dataset = datasets["train"]
    valid_dataset = datasets["validation"]
    # train dataset, train dataloader
    q_seqs = tokenizer(
        train_dataset["question"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    p_seqs = tokenizer(
        train_dataset["context"],
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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    # valid dataset, valid dataloader

    q_seqs = tokenizer(
        valid_dataset["question"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    p_seqs = tokenizer(
        valid_dataset["context"],
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
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)

    p_encoder = BertEncoder.from_pretrained(args.model_name)
    q_encoder = BertEncoder.from_pretrained(args.model_name)

    p_encoder.to(args.device)
    q_encoder.to(args.device)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in p_encoder.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in p_encoder.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
        {
            "params": [
                p
                for n, p in q_encoder.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in q_encoder.named_parameters()
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
    # t_total = len(train_loader) // args.gradient_accumulation_steps * args.num_train_epochs
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=args.warmup_steps,
    #     num_training_steps=t_total
    # )
    cor = []
    ls = []
    for epoch in range(args.epochs):
        step = 0
        losses = 0
        for data in tqdm(train_loader):
            step += 1
            p_inputs = {
                "input_ids": data[0].to(args.device),
                "attention_mask": data[1].to(args.device),
                "token_type_ids": data[2].to(args.device),
            }

            q_inputs = {
                "input_ids": data[3].to(args.device),
                "attention_mask": data[4].to(args.device),
                "token_type_ids": data[5].to(args.device),
            }

            targets = torch.arange(0, len(p_inputs["input_ids"])).long().to(args.device)
            q_output = q_encoder(**q_inputs)
            p_output = p_encoder(**p_inputs)
            retrieval = torch.matmul(q_output, p_output.T)
            retrieval_scores = F.log_softmax(retrieval, dim=1)

            loss = F.nll_loss(retrieval_scores, targets)
            losses += loss.item()
            q_encoder.zero_grad()
            p_encoder.zero_grad()

            loss.backward()
            optimizer.step()
            # scheduler.step()
            if step % 100 == 0:
                print(f"{epoch}epoch loss: {losses/(step)}")
        losses = 0
        correct = 0
        step = 0
        for data in tqdm(valid_loader):
            with torch.no_grad():
                p_inputs = {
                    "input_ids": data[0].to(args.device),
                    "attention_mask": data[1].to(args.device),
                    "token_type_ids": data[2].to(args.device),
                }

                q_inputs = {
                    "input_ids": data[3].to(args.device),
                    "attention_mask": data[4].to(args.device),
                    "token_type_ids": data[5].to(args.device),
                }

                targets = torch.arange(0, args.batch_size).long().to(args.device)
                q_output = q_encoder(**q_inputs)
                p_output = p_encoder(**p_inputs)

                retrieval = torch.matmul(q_output, p_output.T)
                retrieval_scores = F.log_softmax(retrieval, dim=1)
                retrieval_pred = torch.argmax(retrieval_scores, -1)
                # correct += (retrieval_pred == targets).float().sum().to('cpu')
                correct += torch.sum(retrieval_pred == targets).to("cpu")

                loss = F.nll_loss(retrieval_scores, targets)
                losses += loss.item()
                step += 1
        cor.append(correct / (args.batch_size * (step + 1)))
        ls.append(losses / (step + 1))
        print(
            f"{epoch}epoch: loss:{losses/(step)} correct:{correct/(args.batch_size*(step))}"
        )
    phase_vectors = []
    for data in train_dataset:
        p_inputs = {
            "input_ids": data[0].to(args.device),
            "attention_mask": data[1].to(args.device),
            "token_type_ids": data[2].to(args.device),
        }
        output = q_encoder(**p_inputs)
        phase_vectors.append(output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--model_name", type=str, default="klue/bert-base", help="model name"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="../data/train_dataset", help="dataset"
    )
    parser.add_argument("--warmup_steps", type=int, default=500, help="dataset")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="dataset")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="dataset")
    parser.add_argument("--epochs", type=int, default=5, help="epochs")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--batch_size", type=str, default=20, help="batch_size")

    args = parser.parse_args()
    train(args)
