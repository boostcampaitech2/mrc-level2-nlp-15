from elasticsearch import Elasticsearch, helpers
import pandas as pd
from tqdm import tqdm
import json
from datasets import load_from_disk


def get_doc_id(context, wiki):
    for k, v in wiki.items():
        if v["text"] == context:
            return k


class elastic:
    """
    Class for creating ElasticSearch index
    Nori Tokenizer를 활용했으며 Nori POS filter를 통해 query와 passage 사이에 명사와 
    숫자등 중요한 정보를 제외한 token들은 제외시킵니다.
    """

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

        self.index_setting = {
            "settings": {
                "index": {
                    "analysis": {
                        "analyzer": {
                            "my_analyzer": {
                                "type": "custom",
                                "tokenizer": "nori_tokenizer",
                                "decompound_mode": "mixed",
                                "filter": ["shingle", "my_nori_filter"],
                            }
                        },
                        "filter": {
                            "my_nori_filter": {
                                "type": "nori_part_of_speech",
                                "stoptags": [
                                    "E",  ## Verbal Endings
                                    "IC",  ## Interjection
                                    "J",  ## Ending Particle
                                    "MAG",  ## General Adverb
                                    "MAJ",  ## Conjunctive Adverb
                                    "MM",  ## Modifier
                                    "NA",  ## Unknown
                                    "SC",  ## Separator
                                    "SE",  ## Ellipsis
                                    "SF",  ## Terminal punctuation
                                    "SN",  ## Number
                                    "SP",  ## Space
                                    "SSC",  ## Closing brackets
                                    "SSO",  ## Opening brackets
                                    "SY",  ## Other symbol
                                    "UNA",  ## Unknown
                                    "UNKNOWN",  ## Unknown
                                    "VA",  ## Adjectives
                                    "VCN",  ## Negative designator
                                    "VCP",  ## Positive designator
                                    "VSV",  ## Unknown
                                    "VV",  ## Verb
                                    "VX",  ## Auxilary Verb or Adjective
                                    "XPN",  ## Prefix
                                    "XR",  ## Root
                                    "XSA",  ## Adjective Suffix
                                    "XSN",  ## Noun suffix
                                    "XSV",  ## Verb suffix
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
        """
        Constructing ElasticSearch Index
        :return:
        """
        with open(self.context_path) as file:
            wiki = json.load(file)
        search_corpus = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        docs = []
        for i in search_corpus:
            docs.append(
                {
                    "_index": "wikipedia",
                    "_source": {"text": i},
                }
            )

        if self.es.indices.exists(self.index_name):
            pass
        else:
            self.es.indices.create(index=self.index_name, body=self.index_setting)
            helpers.bulk(self.es, docs)

    def retrieve(self, query_or_dataset, topk, wiki):
        """
        Takes in dataset with queries and returns dataframe with information
        about top -k relevant passages
        :param query_or_dataset: dataset with queries
        :param topk: the number of relevant passages to retrieve per question
        :return: dataframe with question and information on the top-k retrieved passages
        """

        datas = []
        for i in tqdm(range(len(query_or_dataset))):
            cp = {i: v for i, v in query_or_dataset[i].items()}

            if "document_id" in cp:

                docu_dict = {
                    "question": cp["question"],
                    "id": cp["id"],
                    "candidate_ids": [],
                    "originial_document_id": cp["document_id"],
                }
            else:
                docu_dict = {
                    "question": cp["question"],
                    "id": cp["id"],
                    "candidate_ids": [],
                }

            if (
                "context" in query_or_dataset[i].keys()
                and "answers" in query_or_dataset[i].keys()
            ):
                cp["original_context"] = query_or_dataset[i]["context"]

            query = query_or_dataset[i]["question"]
            try:
                res = self.es.search(index=self.index_name, q=query, size=topk + 1)
            except:
                query = query.replace("/", "")
                query = query.replace("~", " ")
                res = self.es.search(index=self.index_name, q=query, size=topk + 1)

            x = res["hits"]["hits"]
            context = []
            for docu in x:
                score = docu["_score"]
                doc_id = get_doc_id(docu["_source"]["text"], wiki)
                tup = (doc_id, score)
                context.append(tup)
            context = set(context)
            context = sorted(context, key=lambda x: x[1], reverse=True)
            context = context[:topk]
            docu_dict["candidate_ids"] = context
            datas.append(docu_dict)

        return pd.DataFrame(datas)


if __name__ == "__main__":
    dataset = load_from_disk("./data/train_dataset")
    test_dataset = load_from_disk("./data/test_dataset")
    with open(context_path, "r", encoding="utf-8") as f:
        wiki = json.load(f)

    ### Different datasets to use for retrieving the top related passages
    train_datasets = dataset["train"]
    valid_datasets = dataset["validation"]
    test_datasets = test_dataset["validation"]

    ### Example of usage for validation dataset
    x = elastic("wikipedia")
    x.build_elatic()
    p = x.retrieve(valid_datasets, 100, wiki)
    p.to_csv("./pos_filtered_elastic_top100_val.csv", index=False)
