from elasticsearch import Elasticsearch, helpers
import time
import torch
from torch.nn.functional import softmax
from collections import defaultdict
from datasets import load_from_disk
from tqdm import tqdm
import json
import pandas as pd

class ElasticDPR:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model.cuda()
        self.es = Elastic()
        self.context_path="/opt/ml/data/wikipedia_documents.json"
        with open(self.context_path) as file:
            self.wikipedia = json.load(file)
            print("wikipedia loaded")
        
    def get_doc_scores(self, query, k):
        docs = self.es.retrieve(query=query, k=k)
        contexts = [doc['context'] for doc in docs]
        queries = [query for _ in range(len(docs))] 
        
        tokenized_dataset, sample_id2idx = self.prepare_train_features(queries, contexts)
        
        doc_id2prob = dict()

        for sample_id, idx_list in sample_id2idx.items():
            probs = []
            for idx in idx_list:
                inputs = {key:torch.tensor(val[idx]).unsqueeze(dim=0).cuda() for key,val in tokenized_dataset.items()}
                outputs = self.model(**inputs)
                prob = softmax(outputs.logits.to('cpu'), dim=1).squeeze()[1] # [[0.2,0.8]]->[0.2,0.8]->0.8
                probs.append(float(prob)) # tensor -> float
            doc_id = docs[sample_id]['document_id']
            doc_id2prob[doc_id] = {
                'dpr_prob': sum(probs)/len(probs),
                'es_score': docs[sample_id]['score']
            }
        return doc_id2prob
    
    def prepare_train_features(self, queries, contexts):
        max_seq_length = 384 # 질문과 컨텍스트, special token을 합한 문자열의 최대 길이
        doc_stride = 128 # 컨텍스트가 너무 길어서 나눴을 때 오버랩되는 시퀀스 길이

        tokenized_dataset = self.tokenizer(
                queries,
                contexts,
                truncation="only_second",  # max_seq_length까지 truncate한다. pair의 두번째 파트(context)만 잘라냄.
                max_length=max_seq_length,
                stride=doc_stride,
                return_overflowing_tokens=True, # 길이를 넘어가는 토큰들을 반환할 것인지
                return_offsets_mapping=True,  # 각 토큰에 대해 (char_start, char_end) 정보를 반환한 것인지
                padding="max_length",
            )

        overflow_to_sample_mapping = tokenized_dataset.pop("overflow_to_sample_mapping")
        tokenized_dataset.pop("offset_mapping")
        sample_id2idx = defaultdict(list)
        for i, sample_id in enumerate(overflow_to_sample_mapping):
            sample_id2idx[sample_id].append(i)
        return tokenized_dataset, sample_id2idx
    
    def get_candidate_ids(self, query, es_k, dpr_k):
        assert dpr_k <= es_k, "dpr_k must be less than or equal to es_k"
        
        res = self.get_doc_scores(query=query, k=es_k) 
        ''' 
        res == Dict{
            document_id: Dict{
                'drp_prob': float, 
                'es_score': float
            }
        }
        '''
        candidate_ids = list(res) 
        '''
        candidate_ids == List[document_id:Int]
        '''
        candidate_ids.sort(key=lambda doc_id:res[doc_id]['dpr_prob'],reverse=True)
        
        return candidate_ids[:dpr_k]
    
    def generate_dataframe(self, dataset, es_k, dpr_k):
        is_test = False if 'document_id' in dataset.features else True
        
        if not is_test: # test가 아니면 positive document가 들어있는 확률을 계산
            total, included = 0,0
        
        res = []
        
        for data in tqdm(dataset):
            query = data['question']
            candidate_ids = self.get_candidate_ids(query=query, es_k=es_k, dpr_k=dpr_k)
            
            res.append({
                'question': query,
                'id': data['id'],
                'document_ids': candidate_ids,
                'context': '[SEP]'.join([self.wikipedia[str(i)]['text'].replace('\\n','').replace('\n','') for i in candidate_ids])
            })
            if not is_test:
                document_id = data['document_id']
                if document_id in candidate_ids: included += 1
                total += 1
        if not is_test:     
            print(f"ElasticSearch:{es_k}개 -> DPR:{dpr_k}개 추출 후 정답문서를 포함할 확률 {100*(included/total)}")
        return pd.DataFrame(res)
                


class Elastic:
    def __init__(self, INDEX_NAME="wikipedia", context_path="../data/wikipedia_documents_deduplicated2.json"):
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
                    "document_id": {
                        "type": "integer",
                    },
                }
            },
        }

    def build_elatic(self):
        with open(self.context_path) as file:
            json_data = json.load(file)
        docs = []
        for i, item in json_data.items():
            docs.append(
                {
                    "_id": item["document_id"],
                    "_index": "wikipedia",
                    "_source": {
                        "text": item["text"], 
                        "title": item["title"],
                        "document_id": item["document_id"]
                    },
                }
            )

        if self.es.indices.exists(self.index_name):
            pass
        else:
            self.es.indices.create(index=self.index_name, body=self.index_setting)
            helpers.bulk(self.es, docs)

    def retrieve_df(self, query_or_dataset, topk):
        datas = []
        total = 0
        include_cnt = 0
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
            candidate_docs = res["hits"]["hits"]
            context = []
            candidate_doc_ids = []
            for doc in candidate_docs:
                context.append(doc["_source"]["text"])
                candidate_doc_ids.append(doc["_source"]["document_id"])
            
            cp["context"] = "[SEP]".join(context)
            cp["candidate_doc_ids"] = str(candidate_doc_ids)
            datas.append(cp)

            if cp["document_id"] in candidate_doc_ids:
                include_cnt += 1
            total += 1
        
        print(f"{100*(include_cnt/total)}%의 확률로 ground truth 문서가 포함됨.")
        
        return pd.DataFrame(datas)
    
    def retrieve(self, query, k):
        query = query.replace("/", "").replace("~", " ")
        docs = self.es.search(index=self.index_name, q=query, size=k)["hits"]["hits"]
        
        res = []
        for doc in docs:
            res.append({
                "document_id":doc["_source"]["document_id"],
                "context":doc["_source"]["text"].replace('\\n','').replace('\n',''),
                "score":doc['_score']
            })
        return res
    
    
    def generate_dataset_for_dpr(self, dataset, k):
        data_list = []
        total = 0
        include_cnt = 0
        
        for i in tqdm(range(len(dataset))):
            #item = {key:value for key,value in dataset[i].items()}      
            
            question = dataset[i]['question'].replace("/", "").replace("~", " ")
            document_id = dataset[i]['document_id']
            
            res = self.es.search(index=self.index_name, q=question, size=k)
            docs = res['hits']['hits']
            
            positive_included = False # Flag to check if 'positive context' is included in docs
            candidate_doc_ids = []
            for doc in docs:
                item = dict()
                item['question'] = question
                item['context'] = doc['_source']['text']
                if int(doc['_source']['document_id'])==int(document_id):
                    item['label'] = 1
                    positive_included = True
                else:
                    item['label'] = 0
                    
                data_list.append(item)
                candidate_doc_ids.append(doc["_source"]["document_id"])
            
            if not positive_included: # positive context를 포함하는 데이터를 data_list에 추가
                item = dict()
                item['question'] = question
                item['context'] = dataset[i]['context']
                item['label'] = 1
                data_list.append(item)
            
            if document_id in candidate_doc_ids:
                include_cnt += 1
            total += 1
            
        print(f"{100*(include_cnt/total)}%의 확률로 ground truth 문서가 포함됨.")
        return pd.DataFrame(data_list)