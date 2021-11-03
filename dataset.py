import json
import pandas as pd

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    BertModel, RobertaModel,
    BertPreTrainedModel,
    AdamW, get_linear_schedule_with_warmup,
    TrainingArguments,
)


from typing import List, Dict

class DprDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir,tokenizer):
        dataset_dict = pd.read_csv(data_dir).to_dict('list')
        # 개행문자('\\n','\n') 제거하기
        for i, context in enumerate(dataset_dict['context']):
            dataset_dict['context'][i] = context.replace("\\n","").replace("\n","") 
        self.tokenizer = tokenizer
        self.encodings = self.prepare_train_features(dataset_dict)
        
        
    def __len__(self):
        return len(self.encodings['labels'])
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key,val in self.encodings.items()}
        return item
    
    def prepare_train_features(self, dataset: Dict):
        max_seq_length = 384 # 질문과 컨텍스트, special token을 합한 문자열의 최대 길이
        doc_stride = 128 # 컨텍스트가 너무 길어서 나눴을 때 오버랩되는 시퀀스 길이
        
        tokenized_dataset = self.tokenizer(
                dataset["question"],
                dataset["context"],
                truncation="only_second",  # max_seq_length까지 truncate한다. pair의 두번째 파트(context)만 잘라냄.
                max_length=max_seq_length,
                stride=doc_stride,
                return_overflowing_tokens=True, # 길이를 넘어가는 토큰들을 반환할 것인지
                return_offsets_mapping=True,  # 각 토큰에 대해 (char_start, char_end) 정보를 반환한 것인지
                padding="max_length",
            )

        overflow_to_sample_mapping = tokenized_dataset.pop("overflow_to_sample_mapping")
        tokenized_dataset.pop('offset_mapping')
        labels = []
        for idx in overflow_to_sample_mapping:
            labels.append(dataset['label'][idx])
        tokenized_dataset['labels'] = labels

        return tokenized_dataset