from elasticsearch import Elasticsearch
import pandas as pd
from tqdm import tqdm
import time
import json


class elastic:
    def __init__(self, INDEX_NAME, context_path="../data/wikipedia_documents.json"):
        self.index_name = INDEX_NAME
        try:
            self.es.transport.close()
        except:
            pass
        self.context_path = context_path
        self.es = Elasticsearch()
        self.index_setting = INDEX_SETTINGS = {
            "settings": {
                "index": {
                    "analysis": {
                        "analyzer": {
                            "korean": {
                                "type": "custom",
                                "tokenizer": "nori_tokenizer",
                                "filter": ["shingle"],
                            }
                        }
                    }
                }
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
                }
            },
        }

    def build_elatic(self):
        with open(self.context_path) as file:
            json_data = json.load(file)
        jsones = {}
        for i, j in json_data.items():
            jsones[i] = {"text": j["text"], "title": j["title"]}

        if self.es.indices.exists(self.index_name):
            self.es.indices.delete(index=self.index_name)
        self.es.indices.create(index=self.index_name, body=self.index_setting)

        le = len(jsones.items())
        for doc_id, doc in jsones.items():
            self.es.index(index=self.index_name, id=int(doc_id), body=doc)
            time.sleep(0.1)
            if int(doc_id) % 100 == 0:
                print(100 * int(doc_id) / le)

    def retrieve(self, query_or_dataset, topk):
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
            x = res["hits"]["hits"]
            context = []
            for docu in x:
                context.append(docu["_source"]["text"])
            cp["context"] = "///".join(context)
            datas.append(cp)

        return pd.DataFrame(datas)


if __name__ == "__main__":
    from datasets import load_from_disk

    dataset = load_from_disk("../data/train_dataset")
    datasets = dataset["validation"]
    x = elastic("toy_index")
    x.build_elatic()
    p = x.retrieve(datasets, 3)
    print(p.head())
