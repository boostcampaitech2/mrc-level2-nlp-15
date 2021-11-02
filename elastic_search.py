from elasticsearch import Elasticsearch, helpers
import pandas as pd
from tqdm import tqdm
import time
import json
from typing import List, Tuple, NoReturn, Any, Optional, Union
import random
import re


def get_doc_id(context, wiki):
    for k, v in wiki.items():
        if v["text"] == context:
            return k


class elastic:
    def __init__(self, INDEX_NAME, context_path="../data/wikipedia_documents.json"):
        self.index_name = INDEX_NAME
        try:
            self.es.transport.close()
        except:
            pass
        self.context_path = context_path
        config = {
            "host": "localhost",
            "port": 9200,
            "timeout": 100,
            "max_retries": 10,
            "retry_on_timeout": True,
        }
        self.es = Elasticsearch([config])
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
                    "text": {"type": "text", "analyzer": "korean", "search_analyzer": "korean"},
                    "title": {"type": "text", "analyzer": "korean", "search_analyzer": "korean"},
                    # "document_id": {
                    #     "type": "long",
                    # },
                }
            },
        }
        with open(self.context_path) as file:
            self.json_data = json.load(file)

    def build_elatic(self):
        docs = []
        for i, j in self.json_data.items():
            docs.append(
                {
                    "_index": "wikipedia",
                    "_source": {
                        "text": j["text"],
                        "title": j["title"],
                        # "document_id": j["document_id"],
                    },
                }
            )

        if self.es.indices.exists(self.index_name):
            pass
        else:
            self.es.indices.create(index=self.index_name, body=self.index_setting)
            helpers.bulk(self.es, docs)

    def retrieve(self, query_or_dataset, topk):
        datas = []
        for i in tqdm(range(len(query_or_dataset))):
            cp = {i: v for i, v in query_or_dataset[i].items()}
            if "context" in query_or_dataset[i].keys() and "answers" in query_or_dataset[i].keys():
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
                document_id.append(get_doc_id(docu["_source"]["text"], self.json_data))
            # score = list(map(lambda x: str(x/sum(score)),score))
            cp["context"] = "///".join(context)  # 리스트를 사용하려면 join없이 그냥 context를 쓰면 됩니다.
            cp["score"] = score
            cp["document_id"] = document_id
            datas.append(cp)

        return pd.DataFrame(datas)

    def retrieve_for_train(self, query_or_dataset, topk):
        datas = []
        j = 0
        for i in tqdm(range(len(query_or_dataset))):
            cp = {i: v for i, v in query_or_dataset[i].items()}
            if "context" in query_or_dataset[i].keys() and "answers" in query_or_dataset[i].keys():
                original_context = query_or_dataset[i]["context"]

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
            if original_context not in context:
                x = random.randint(0, 4)
                context[x] = original_context
            else:
                x = context.index(original_context)
                context[x] = original_context
            answer_start = cp["answers"]["answer_start"][0]
            for i, j in enumerate(context):
                if j == original_context:
                    break
                else:
                    answer_start += len(j)
            cp["context"] = "".join(context)  # 리스트를 사용하려면 join없이 그냥 context를 쓰면 됩니다.
            cp["score"] = score
            cp["document_id"] = document_id
            cp["answers"]["answer_start"] = [answer_start]
            datas.append(cp)
        return pd.DataFrame(datas)

    def get_relevant_doc(self, query: str, topk: Optional[int] = 1) -> Tuple[List, List]:
        query = query.replace("/", "")
        query = query.replace("~", " ")
        res = self.es.search(index=self.index_name, q=query, size=topk)
        hits = res["hits"]["hits"]
        context = []
        for docu in hits:
            context.append(docu["_source"]["text"])
        join_context = "///".join(context)
        return join_context


if __name__ == "__main__":
    from datasets import load_from_disk

    dataset = load_from_disk("../data/train_dataset")
    datasets = dataset["validation"]
    x = elastic("wikipedia")
    x.build_elatic()
    p = x.retrieve(datasets, 100)
    # p.to_csv('top100_wikipedia.csv')
