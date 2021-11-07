# import python modules
import re
import time
import json
from typing import List, Tuple, NoReturn, Any, Optional, Union
import random

# import data wrangling modules
import pandas as pd
from tqdm import tqdm

# import torch and its related modules
import torch
import torch.nn.functional as F

# import third party modules
from elasticsearch import Elasticsearch, helpers


class elastic:
    def __init__(self,index_name:str,context_path:str="../data/wikipedia_documents.json") -> None:
        self.index_name = index_name
        try:
            self.es.transport.close()
        except:
            pass
        self.context_path = context_path
        self.es = Elasticsearch(timeout=100, max_retries=10, retry_on_timeout=True)
        self.index_setting = {
            "settings": {
                "index": {
                    "analysis": {
                        "analyzer": {
                            "korean": {
                                "type": "custom",
                                "decompound_mode": "mixed",
                                "tokenizer": "nori_tokenizer",
                                "filter": ["shingle"],
                            }
                        },
                    },
                },
            },
            "mappings": {
                "properties": {
                    "text": {
                        "type": "text",
                        "analyzer": "korean",
                        "search_analyzer": "korean",
                    },
                    "title": {
                        "type": "text",
                        "analyzer": "korean",
                        "search_analyzer": "korean",
                    },
                    "document_id": {
                        "type": "long",
                    },
                }
            },
        }

    def build_elatic(self) -> None:
        """elastic search index가 있는지 확인하고 없으면 만드는 부분입니다."""
        with open(self.context_path) as file:
            json_data = json.load(file)
        docs = []
        for i, item in json_data.items():
            docs.append(
                {
                    "_index": "wikipedia",
                    "_source": {
                        "text": item["text"],
                        "title": item["title"],
                        "document_id": item["document_id"],
                    },
                }
            )

        if self.es.indices.exists(self.index_name):
            pass
        else:
            self.es.indices.create(index=self.index_name, body=self.index_setting)
            helpers.bulk(self.es, docs)

    def retrieve(self,dataset:Dataset,topk:int)->pd.DataFrame:
        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional):
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """
        datas = []
        for i in tqdm(range(len(query_or_dataset))):
            cp = {i: v for i, v in query_or_dataset[i].items()}
            if (
                "context" in query_or_dataset[i].keys()
                and "answers" in query_or_dataset[i].keys()
            ):
                cp["original_context"] = query_or_dataset[i]["context"]

            query = query_or_dataset[i]["question"]
            query = query.replace("/", "")
            query = query.replace("~", " ")
            res = self.es.search(index=self.index_name, q=query, size=topk)
            hits = res["hits"]["hits"]
            context = []
            score = []
            document_id = []
            for docu in hits:
                context.append(docu["_source"]["text"])
                score.append(docu["_score"])
                document_id.append(docu["_source"]["document_id"])

            cp["context"] = "///".join(context)  
            cp["score"] = score
            cp["document_id"] = document_id
            datas.append(cp)

        return pd.DataFrame(datas)

    def get_relevant_doc(
        self, query: str, topk: Optional[int] = 1
    ) -> Tuple[List, List]:
        query = query.replace("/", "")
        query = query.replace("~", " ")
        res = self.es.search(index=self.index_name, q=query, size=topk)
        hits = res["hits"]["hits"]
        context = []
        for docu in hits:
            context.append(docu["_source"]["text"])
        join_context = "<다음 문맥>".join(context)
        return join_context

    def get_relevant_doc(self, query: str, topk: Optional[int] = 1) -> str:
        """
        Arguments:
            query str:
                질문에 해당하는 str을 받습니다.
            topk (Optional[int], optional):
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            topk개의 문서가 연결된 str을 반환합니다.

        Note:
            하나의 문장에 대한 입력만 받을 수 있습니다.
        """
        docs = self.es.retrieve(query=query, k=k)
        contexts = [doc["context"] for doc in docs]
        queries = [query for _ in range(len(docs))]

        tokenized_dataset, sample_id2idx = self.prepare_train_features(
            queries, contexts
        )

        doc_id2prob = dict()

        for sample_id, idx_list in sample_id2idx.items():
            probs = []
            for idx in idx_list:
                inputs = {
                    key: torch.tensor(val[idx]).unsqueeze(dim=0).cuda()
                    for key, val in tokenized_dataset.items()
                }
                outputs = self.model(**inputs)
                prob = F.softmax(outputs.logits.to("cpu"), dim=1).squeeze()[
                    1
                ]  # [[0.2,0.8]]->[0.2,0.8]->0.8
                probs.append(float(prob))  # tensor -> float
            doc_id = docs[sample_id]["document_id"]
            doc_id2prob[doc_id] = {
                "dpr_prob": sum(probs) / len(probs),
                "es_score": docs[sample_id]["score"],
            }
        return doc_id2prob


if __name__ == "__main__":
    from datasets import load_from_disk

    dataset = load_from_disk("../data/train_dataset")
    datasets = dataset["validation"]
    x = elastic("wikipedia")
    x.build_elatic()
    p = x.retrieve(datasets, 100)
    # p.to_csv('top100_wikipedia.csv')
