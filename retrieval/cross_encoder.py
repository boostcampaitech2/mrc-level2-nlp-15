import torch
import random
import numpy as np
import torch.nn.functional as F
import argparse

from tqdm import trange
from tqdm.auto import tqdm

from transformers import (
    AutoTokenizer,
    RobertaPreTrainedModel,
    RobertaModel,
    BertModel,
    BertPreTrainedModel,
    AdamW,
    get_linear_schedule_with_warmup,
    TrainingArguments,
)

from datasets import load_from_dist, DatasetDict

from torch.utils.data import Sampler, TensorDataset, DataLoader


def set_seed(random_seed):
    """
    Random number fixed
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    random.seed(random_seed)
    np.random.seed(random_seed)


class BertEncoder(BertPreTrainedModel):
    """
    Encoder using BertModel as a backbone model
    """

    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)  # Call BertModel
        self.init_weights()  # initalized Weight
        classifier_dropout = (  # Dropout
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = torch.nn.Dropout(classifier_dropout)
        self.linear = torch.nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,  # If you want to use Roberta Model, Comment out this code
        )

        pooled_output = outputs[1]  # CLS pooled output
        pooled_output = self.dropout(pooled_output)  # apply dropout
        output = self.linear(pooled_output)  # apply classifier
        return output


class RoBertaEncoder(RobertaPreTrainedModel):
    """
    Encoder using RoberatModel as a backbone model
    """

    def __init__(self, config):
        super(RoBertaEncoder, self).__init__(config)

        self.roberta = RobertaModel(config)  # Call RobertaModel
        self.init_weights()  # initalized Weight
        classifier_dropout = (  # Dropout
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = torch.nn.Dropout(classifier_dropout)
        self.linear = torch.nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        # token_type_ids=None
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        output = self.linear(pooled_output)
        return output


class CustomSampler(Sampler):
    """
    When creating a DataLoader, make sure
    that three consecutive indexes do not included in one batch

    This CustomSampler assumes that one q-p pair is split into three.
    If it splits more,
    you need to modify "abs(s-f) <= 2" in the code below to fit the length

    you don't have to use this code
    But, if you don't use this code, you have to insert 'shuffle=True' in your DataLoader
    """

    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

    def __iter__(self):
        n = len(self.data_source)
        index_list = []
        while True:
            out = True
            for i in range(self.batch_size):  # Creat an index list of batch_size
                tmp_data = random.randint(0, n - 1)
                index_list.append(tmp_data)
            for f, s in zip(index_list, index_list[1:]):
                if (
                    abs(s - f) <= 2
                ):  # If splits more, modify this code like 'abs(s-f) <= 3'
                    out = False
            if out == True:
                break

        while True:  # Insert additional index data according to condition and length
            tmp_data = random.randint(0, n - 1)
            if (tmp_data not in index_list) and (
                abs(tmp_data - index_list[-i])
                > 2  # If splits more, modify this code like 'abs(tmp_data - index_list[-i]) > 3'
                for i in range(1, self.batch_size + 1)
            ):
                index_list.append(tmp_data)
            if len(index_list) == n:
                break
        return iter(index_list)

    def __len__(self):
        return len(self.data_source)


def train(
    args,
    dataset: DatasetDict,
    tokenizer,
    cross_encoder,
    sampler=None,
):
    """
    In-batch Negative CrossEncoder Train

    Arg:
        dataset: DatasetDict
            it has 'question' and 'context'
            if your data does not match this code, you only need to match
             it in the form of getting question and context
        tokenizer
        sampler: Sampler
            you can use the CustomSampler
            if you don't want to use CustomSampler,
             you have to insert 'shuffle=True' in your DataLoader
    """
    tokenized_examples = tokenizer(
        dataset["question"],
        dataset["context"],
        truncation="only_second",
        max_length=512,
        strid=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_token_type_ids=False,  # if you want to use bert tokenizer, you have to be marked as 'True'
        padding="max_length",
        return_tensors="pt",
    )

    train_dataset = TensorDataset(
        tokenized_examples["input_ids"],
        tokenized_examples["attention_mask"],
        # tokenized_examples['token_type_ids] # When you use BertModel, release annotation
    )

    if sampler is not None:
        sampler = sampler(train_dataset, args.per_device_train_batch_size)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.per_device_train_batch_size,
            sampler=sampler,
            drop_last=True,
        )
    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.per_device_train_batch_size,
            shuffle=True,
            drop_last=True,
        )

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in cross_encoder.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in cross_encoder.named_parameters()
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

    cross_encoder.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    cross_encoder.train()

    for epoch, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        losses = 0
        for step, batch in enumerate(epoch_iterator):
            # if torch.cuda.is_available() :
            #     batch = tuple(t.cuda() for t in batch)

            cross_inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                # 'token_type_ids' : batch[2] # When you use BertModel, release annotation
            }
            for k in cross_inputs.keys():
                cross_inputs[k] = cross_inputs[k].tolist()

            # Make In-Batch Negative Sampling
            new_input_ids = []
            new_attention_mask = []
            # new_token_type_ids = [] # When you use BertModel, release annotation
            for i in range(len(cross_inputs["input_ids"])):
                sep_index = cross_inputs["input_ids"][i].index(
                    tokenizer.sep_token_id
                )  # [SEP] token의 index

                for j in range(len(cross_inputs["input_ids"])):
                    query_id = cross_inputs["input_ids"][i][:sep_index]
                    query_att = cross_inputs["attention_mask"][i][:sep_index]
                    # query_tok = cross_inputs['token_type_ids'][i][:sep_index] # When you use BertModel, release annotation

                    context_id = cross_inputs["input_ids"][j][sep_index:]
                    context_att = cross_inputs["attention_mask"][j][sep_index:]
                    # context_tok = cross_inputs['token_type_ids'][j][sep_index:] # When you use BertModel, release annotation
                    query_id.extend(context_id)
                    query_att.extend(context_att)
                    # query_tok.extend(context_tok) # When you use BertModel, release annotation
                    new_input_ids.append(query_id)
                    new_attention_mask.append(query_att)
                    # new_token_type_ids.append(query_tok) # When you use BertModel, release annotation

            new_input_ids = torch.tensor(new_input_ids)
            new_attention_mask = torch.tensor(new_attention_mask)
            # new_token_type_ids = torch.tensor(new_token_type_ids) # When you use BertModel, release annotation
            if torch.cuda.is_available():
                new_input_ids = new_input_ids.to("cuda")
                new_attention_mask = new_attention_mask.to("cuda")
                # new_attention_mask = new_attention_mask.to('cuda') # When you use BertModel, release annotation

            change_cross_inputs = {
                "input_ids": new_input_ids,
                "attention_mask": new_attention_mask,
                #'token_type_ids' : new_token_type_ids # When you use BertModel, release annotation
            }

            cross_output = cross_encoder(**change_cross_inputs)
            cross_output = cross_output.view(-1, args.per_device_train_batch_size)
            targets = torch.arange(0, args.per_device_train_batch_size).long()

            if torch.cuda.is_available():
                targets = targets.to("cuda")

            score = F.log_softmax(cross_output, dim=1)
            loss = F.nll_loss(score, targets)
            #########################No ACCUMULATION#########################
            # losses += loss.item()
            # if step % 100 == 0 :
            #     print(f'{epoch}epoch loss: {losses/(step+1)}') # Accumulation할 경우 주석처리

            # self.cross_encoder.zero_grad()
            # loss.backward()
            # optimizer.step()
            # scheduler.step()
            #################################################################

            #############################ACCUMULATION#########################
            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                cross_encoder.zero_grad()

            losses += loss.item()
            if (step + 1) % 100 == 0:
                train_loss = losses / 100
                print(f"training loss: {train_loss:4.4}")
                losses = 0
            ##################################################################

    return cross_encoder


if __name__ == "__main__":
    args = TrainingArguments(
        output_dir="your_output_directory",  # put in your output directory
        evaluation_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=20,
        weight_decay=0.01,
    )

    set_seed(42)  # magic number :)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = load_from_dist(
        "/opt/ml/data/train_dataset"
    )  # dataset have train/valid dataset

    train_dataset = dataset["train"]

    # you can use 'klue/bert-base' model, and you have to change the code above.
    model_checkpoint = "klue/roberta-large"

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    cross_encoder = RoBertaEncoder.from_pretrained(model_checkpoint)
    if torch.cuda.is_available():
        cross_encoder = cross_encoder.to("cuda")

    sampler = CustomSampler
    c_encoder = train(
        args,
        train_dataset,
        tokenizer,
        cross_encoder,
        sampler=sampler,  # you don't have to use sampler
    )

    torch.save(c_encoder, "/your_save_directory/c_encoder.pt")
