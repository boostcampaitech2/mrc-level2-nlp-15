from elasticsearch import Elasticsearch, helpers
import pandas as pd
from tqdm import tqdm
import json


class elastic:
    def __init__(self, INDEX_NAME, context_path="./data/wikipedia_documents.json"):
        self.index_name = INDEX_NAME
        try:
            self.es.transport.close()
        except:
            pass
        self.context_path = context_path
        config = {
            u"host": u"localhost",
            u"port": b"9200",
            "timeout": 100,
            "max_retries": 10,
            "retry_on_timeout": True,
        }
        self.es = Elasticsearch([config])

        #### POS list: https://lucene.apache.org/core/8_9_0/analyzers-nori/org/apache/lucene/analysis/ko/POS.Tag.html
        ### Paper: http://hiai.co.kr/wp-content/uploads/2020/10/%E1%84%8B%E1%85%B5%E1%84%90%E1%85%A2%E1%84%86%E1%85%B5%E1%86%AB_%E1%84%80%E1%85%B5%E1%84%80%E1%85%A8-%E1%84%83%E1%85%A9%E1%86%A8%E1%84%92%E1%85%A2%E1%84%85%E1%85%B3%E1%86%AF-%E1%84%8B%E1%85%B5%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB-COVID-19-%E1%84%82%E1%85%B2%E1%84%89%E1%85%B3-%E1%84%83%E1%85%A9%E1%84%86%E1%85%A6%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%8B%E1%85%B4-%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%AE%E1%86%A8%E1%84%8B%E1%85%A5-%E1%84%8C%E1%85%B5%E1%86%AF%E1%84%8B%E1%85%B4%E1%84%8B%E1%85%B3%E1%86%BC%E1%84%83%E1%85%A1%E1%86%B8-%E1%84%8E%E1%85%A2%E1%86%BA%E1%84%87%E1%85%A9%E1%86%BA.pdf
        self.index_setting = {
            "settings": {
                "index": {
                    "analysis": {
                        "analyzer": {
                            "my_analyzer": {
                                "type": "custom",
                                "tokenizer": "nori_tokenizer",
                                "filter": ["my_nori_filter"],
                            }
                        },
                        "filter": {
                            "my_nori_filter": {
                                "type": "nori_part_of_speech",
                                "stoptags": [
                                    "NR",  # numeral (하나 밖에 )
                                    "NA",  # Unknown
                                    "SC",  # Separator
                                    "SE",  # Ellipsis (줄임표)
                                    "SF",  # Terminal punctuation (물음표, 느낌표)
                                    "SP",  # Space (공백)
                                    "UNA",  # Unknown
                                    "UNKNOWN",  # Unknwon
                                    "E",  # Verbal Endings (사랑/하(E)/다
                                    "J",  # Ending Particle/조사: (나/는(J)
                                    "IC",  # Interjection 감탄사, (와우, 맙소사)
                                    "MAJ",  # 접속 부사  (그런데, 그러나)
                                    "SH",  # Chinese Character
                                    "VA",  # Adjective 형용사
                                    "VX",  # 보조 용언 (가지/고/싶(VX)/다)
                                    "VV",  # Verb
                                    "SY",  # Other symbol
                                    "MAG",  # General Adverb (빨리, 과연),
                                    "MAJ",  # Conjunctive Adverb,
                                    "SL",  # Foreign Language
                                    "SSO",
                                    "SSC",
                                    "VCN",
                                    "VCP",
                                    "VSV",  # Negative designator
                                    "XPN",  # Positive designator
                                    "XSN",  # Noun suffix
                                    "XSV",  # Verb suffix
                                    "MM",
                                    "SL",
                                    "SN",
                                    "XR",
                                ],
                            }
                        },
                    }
                }
            },
            "mappings": {
                "properties": {
                    "text": {
                        "type": "text",
                        "analyzer": "my_analyzer",
                        "search_analyzer": "my_analyzer",
                    },
                    "title": {
                        "type": "text",
                        "analyzer": "my_analyzer",
                        "search_analyzer": "my_analyzer",
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
            if (
                "context" in query_or_dataset[i].keys()
                and "answers" in query_or_dataset[i].keys()
            ):
                cp["original_context"] = query_or_dataset[i]["context"]

            query = query_or_dataset[i]["question"]
            query = query.replace("/", "")
            query = query.replace("~", " ")
            # res = es.search(index= INDEX_NAME, analyzer = "korean", q = query)
            res = self.es.search(index=self.index_name, q=query, size=topk + 1)
            x = res["hits"]["hits"]
            context = []
            for docu in x:
                context.append(docu["_source"]["text"])
            cp["context"] = "///".join(context)
            datas.append(cp)

        return pd.DataFrame(datas)

    def retrieve_false(self, query_or_dataset, topk):
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
            res = self.es.search(index=self.index_name, q=query, size=topk + 1)
            x = res["hits"]["hits"]
            context = []
            for docu in x:
                cont = docu["_source"]["text"]
                # groud truth 제거
                if cont == cp["original_context"]:
                    continue
                context.append(docu["_source"]["text"])
            context = context[:topk]
            cp["context"] = "///".join(context)
            datas.append(cp)

        return pd.DataFrame(datas)


from datasets import load_from_disk

dataset = load_from_disk("./data/train_dataset")
train_datasets = dataset["train"]
valid_datasets = dataset["validation"]

x = elastic("wikipedia")
x.build_elatic()
p = x.retrieve(valid_datasets, 10)
