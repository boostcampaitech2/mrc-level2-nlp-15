from elasticsearch import Elasticsearch, helpers
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
        self.es = Elasticsearch(timeout=100, max_retries=10, retry_on_timeout=True)
        self.index_setting = {
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
        docs = []
        for i, j in json_data.items():
            docs.append(
                {
                    "_index": "wikipedia",
                    "_source": {"text": j["text"], "title": j["title"]},
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
    x = elastic("wikipedia")
    x.build_elatic()
    p = x.retrieve(datasets, 3)
    print(p.head())
