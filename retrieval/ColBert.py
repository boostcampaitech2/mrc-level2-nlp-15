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


class QueryTokenizer:
    def __init__(self):
        self.tok = BertTokenizerFast.from_pretrained("klue/bert-base")

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
            prefix
            + lst
            + suffix
            + [self.mask_token] * (self.query_maxlen - (len(lst) + 3))
            for lst in tokens
        ]

        return tokens

    def encode(self, batch_text, add_special_tokens=False):
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
    def __init__(self):
        self.tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

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
            ids, mask, reverse_indices = _sort_by_length(ids, mask, bsize)
            batches = _split_into_batches(ids, mask, bsize)
            return batches, reverse_indices

        return ids, mask


def tensorize_triples(
    query_tokenizer, doc_tokenizer, queries, positives, negatives, bsize
):
    # assert len(queries) == len(positives) == len(negatives)
    # assert bsize is None or len(queries) % bsize == 0

    N = len(queries)
    queries = queries.to_list()
    # print(f'queries-----------------------------------------------------')
    # print(queries)
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


def print_message(*s, condition=True):
    s = " ".join([str(x) for x in s])
    msg = "[{}] {}".format(datetime.datetime.now().strftime("%b %d, %H:%M:%S"), s)

    if condition:
        print(msg, flush=True)

    return msg


def timestamp():
    format_str = "%Y-%m-%d_%H.%M.%S"
    result = datetime.datetime.now().strftime(format_str)
    return result


def file_tqdm(file):
    print(f"#> Reading {file.name}")

    with tqdm.tqdm(
        total=os.path.getsize(file.name) / 1024.0 / 1024.0, unit="MiB"
    ) as pbar:
        for line in file:
            yield line
            pbar.update(len(line) / 1024.0 / 1024.0)

        pbar.close()


def save_checkpoint(path, epoch_idx, mb_idx, model, optimizer, arguments=None):
    print(f"#> Saving a checkpoint to {path} ..")

    if hasattr(model, "module"):
        model = model.module  # extract model from a distributed/data-parallel wrapper

    checkpoint = {}
    checkpoint["epoch"] = epoch_idx
    checkpoint["batch"] = mb_idx
    checkpoint["model_state_dict"] = model.state_dict()
    checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    checkpoint["arguments"] = arguments

    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, do_print=True):
    if do_print:
        print_message("#> Loading checkpoint", path, "..")

    if path.startswith("http:") or path.startswith("https:"):
        checkpoint = torch.hub.load_state_dict_from_url(path, map_location="cpu")
    else:
        checkpoint = torch.load(path, map_location="cpu")

    state_dict = checkpoint["model_state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if k[:7] == "module.":
            name = k[7:]
        new_state_dict[name] = v

    checkpoint["model_state_dict"] = new_state_dict

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except:
        print_message("[WARNING] Loading checkpoint with strict=False")
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if do_print:
        print_message("#> checkpoint['epoch'] =", checkpoint["epoch"])
        print_message("#> checkpoint['batch'] =", checkpoint["batch"])

    return checkpoint


def create_directory(path):
    if os.path.exists(path):
        print("\n")
        print_message("#> Note: Output directory", path, "already exists\n\n")
    else:
        print("\n")
        print_message("#> Creating directory", path, "\n\n")
        os.makedirs(path)


def f7(seq):
    """
    Source: https://stackoverflow.com/a/480227/1493011
    """

    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


def batch(group, bsize, provide_offset=False):
    offset = 0
    while offset < len(group):
        L = group[offset : offset + bsize]
        yield ((offset, L) if provide_offset else L)
        offset += len(L)
    return


class dotdict(dict):
    """
    dot.notation access to dictionary attributes
    Credit: derek73 @ https://stackoverflow.com/questions/2352181
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def flatten(L):
    return [x for y in L for x in y]


def zipstar(L, lazy=False):
    """
    A much faster A, B, C = zip(*[(a, b, c), (a, b, c), ...])
    May return lists or tuples.
    """

    if len(L) == 0:
        return L

    width = len(L[0])

    if width < 100:
        return [[elem[idx] for elem in L] for idx in range(width)]

    L = zip(*L)

    return L if lazy else list(L)


def zip_first(L1, L2):
    length = len(L1) if type(L1) in [tuple, list] else None

    L3 = list(zip(L1, L2))

    assert length in [None, len(L3)], "zip_first() failure: length differs!"

    return L3


def int_or_float(val):
    if "." in val:
        return float(val)

    return int(val)


def load_ranking(path, types=None, lazy=False):
    print_message(f"#> Loading the ranked lists from {path} ..")

    try:
        lists = torch.load(path)
        lists = zipstar([l.tolist() for l in tqdm.tqdm(lists)], lazy=lazy)
    except:
        if types is None:
            types = itertools.cycle([int_or_float])

        with open(path) as f:
            lists = [
                [typ(x) for typ, x in zip_first(types, line.strip().split("\t"))]
                for line in file_tqdm(f)
            ]

    return lists


def save_ranking(ranking, path):
    lists = zipstar(ranking)
    lists = [torch.tensor(l) for l in lists]

    torch.save(lists, path)

    return lists


def groupby_first_item(lst):
    groups = defaultdict(list)

    for first, *rest in lst:
        rest = rest[0] if len(rest) == 1 else rest
        groups[first].append(rest)

    return groups


def process_grouped_by_first_item(lst):
    """
        Requires items in list to already be grouped by first item.
    """

    groups = defaultdict(list)

    started = False
    last_group = None

    for first, *rest in lst:
        rest = rest[0] if len(rest) == 1 else rest

        if started and first != last_group:
            yield (last_group, groups[last_group])
            assert (
                first not in groups
            ), f"{first} seen earlier --- violates precondition."

        groups[first].append(rest)

        last_group = first
        started = True

    return groups


def grouper(iterable, n, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks
        Example: grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
        Source: https://docs.python.org/3/library/itertools.html#itertools-recipes
    """

    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


# see https://stackoverflow.com/a/45187287
class NullContextManager(object):
    def __init__(self, dummy_resource=None):
        self.dummy_resource = dummy_resource

    def __enter__(self):
        return self.dummy_resource

    def __exit__(self, *args):
        pass


def load_batch_backgrounds(args, qids):
    if args.qid2backgrounds is None:
        return None

    qbackgrounds = []

    for qid in qids:
        back = args.qid2backgrounds[qid]

        if len(back) and type(back[0]) == int:
            x = [args.collection[pid] for pid in back]
        else:
            x = [args.collectionX.get(pid, "") for pid in back]

        x = " [SEP] ".join(x)
        qbackgrounds.append(x)

    return qbackgrounds


class ColBERT(BertPreTrainedModel):
    def __init__(
        self,
        config,
        mask_punctuation=string.punctuation,
        dim=128,
        similarity_metric="cosine",
    ):
        super(ColBERT, self).__init__(config)

        self.similarity_metric = similarity_metric
        self.dim = dim

        self.mask_punctuation = mask_punctuation
        self.skiplist = {}

        if self.mask_punctuation:
            self.tokenizer = BertTokenizerFast.from_pretrained("klue/bert-base")
            self.skiplist = {
                w: True
                for symbol in string.punctuation
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

    def query(self, input_ids, attention_mask=None, token_type_ids=None):
        # input_ids, attention_mask = input_ids.to("cuda"), attention_mask.to("cuda")
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        # Q_pooled_outputs = Q_outputs[1]
        Q = self.linear(Q)
        # return Q
        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def doc(self, input_ids, attention_mask=None, token_type_ids=None):
        # input_ids, attention_mask = input_ids.to("cuda"), attention_mask.to("cuda")
        D = self.bert(input_ids, attention_mask=attention_mask)[0]
        D = self.linear(D)

        mask = torch.tensor(self.mask(input_ids), device="cuda").unsqueeze(2).float()
        D = D * mask

        D = torch.nn.functional.normalize(D, p=2, dim=2)

        # if not keep_dims:
        #     D, mask = D.cpu().to(dtype=torch.float16), mask.cpu().bool().squeeze(-1)
        #     D = [d[mask[idx]] for idx, d in enumerate(D)]

        return D

    def score(self, Q, D):
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
        """
        학습과 추론에 사용될 여러 셋업을 마쳐봅시다.
        """

        self.args = args
        self.dataset = dataset

        self.q_tokenizer = q_tokenizer
        self.d_tokenizer = d_tokenizer
        self.colbert_encoder = colbert_encoder

        self.df = df

    def train(self, args=None, df=None):
        if args is None:
            args = self.args

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

        for epoch, _ in enumerate(train_iterator):
            losses = 0
            for step, batch in enumerate(train_dataloader):
                # if torch.cuda.ise(t.to_device('cuda') for t in batch)

                # D
                p_inputs = {
                    "input_ids": batch[0].cuda(),
                    "attention_mask": batch[1].cuda(),
                    "token_type_ids": batch[2].cuda(),
                }
                # Q
                q_inputs = {
                    "input_ids": batch[3].cuda(),
                    "attention_mask": batch[4].cuda(),
                    "token_type_ids": batch[5].cuda(),
                }

                sim_scores = self.colbert_encoder(Q=q_inputs, D=p_inputs)

                # Calculate similarity score & loss
                print(f"sim_scores {sim_scores}")

                # target: position of positive samples = diagonal element
                targets = torch.zeros(args.per_device_train_batch_size).long()

                if torch.cuda.is_available():
                    targets = targets.to("cuda")

                print(
                    f"sim_scores.shape, targets.shate {sim_scores.shape} | {targets.shape}"
                )

                loss = criterion(sim_scores.size(1), targets.size(1))
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

                del p_inputs, q_inputs

        return self.colbert_encoder


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
    q_tokenizer = QueryTokenizer()
    d_tokenizer = DocTokenizer()

    # load dataframe(es negative) !!!
    df = pd.read_csv("/opt/ml/data/colbertdata_join_top10_wikipedia.csv")

    retriever = ColBertRetrieval(
        args=args,
        dataset=train_dataset,
        q_tokenizer=q_tokenizer,
        d_tokenizer=d_tokenizer,
        colbert_encoder=colbert_encoder,
        df=df,
    )

    # if torch.cuda.is_available():
    #     p_encoder.cuda()
    #     q_encoder.cuda()

    colbert_encoder = retriever.train()


if __name__ == "__main__":
    dataset_path = '../dataset/train_dataset/"'
    seed()
    main(dataset_path)
