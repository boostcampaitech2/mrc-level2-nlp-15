# Todo for Two Weeks



## Dataset Annotation

Text Generation하는 걸 이용해서 거기에다가 question을 생성해서 train을 할 수 있게 만드는 것도 올라왔다. 

- Mecab -> POS Tagging: 안영진. Answer POS Tagging만 할 것인지, 아니면 전체 context에다에다가 할 것인지는 고민할 부분.
- [Khaii -> POS Tagging](https://github.com/kakao/khaiii)
- [Question Generation](https://kakaobrain.github.io/pororo/seq2seq/qg.html) / [청계산셰르파팀이 했던 KoGPT-2 기반 Question Generation](https://stages.ai/competitions/77/discussion/talk/post/769): 최성욱



## Reader

- [ ] Masked Training with klue/roberta-large
- [ ] [Pretraining method with REALM](https://arxiv.org/pdf/2002.08909.pdf) : Retrieval-Augmented Language Model Pre-Training: Salient Mask pretraining method on spanbert.
- [ ] [PORORO MRC Reader BrainRoberta model](https://kakaobrain.github.io/pororo/tagging/mrc.html): 최성욱



## Retrieval

- [ ] [Elasticsearch-py](https://github.com/elastic/elasticsearch-py): 전재영, 남세현
- [ ] 베이스라인에 있는 TFIDF 대신에 BM25 시도. 다만 BM25 종류가 [okapi-bm25 외에도 가지각각인 듯](https://github.com/dorianbrown/rank_bm25): 김준홍
- [ ] [Dense Passage Retriever(DPR)](https://github.com/deepset-ai/haystack/blob/master/tutorials/Tutorial6_Better_Retrieval_via_DPR.ipynb) : Trainset에서 wiki에서 잘 찾아오는지 그걸 해보면서 이해하는 중: 김현수, 남세현
- [ ] [ColBERT 2020](https://github.com/stanford-futuredata/ColBERT)



## Closed Book QA

- [ ] [Ke-T5](https://github.com/AIRC-KETI/ke-t5): 안영진



---

## Ensemble Method

- Retrieval Method에서 ensemble을 해야 하나?
- Classification 단계에서 ensemble을 해야 하나? 근데 Generation이 들어가면 어떻게 될 지 잘 모르겠어요. 
  - 정답 단어를 가지고 hard voting을 해야 할 것 같음
  - 아니면 정답을 random으로 뽑게 ㅋㅋㅋㅋㅋㅋ