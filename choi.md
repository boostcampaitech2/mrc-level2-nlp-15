### 실험
1. Pororo Library의 MRC활용
2. Question Generation을 이용한 Augmentation

### 폴더
1. PororoMRC: main branch -> inference.md
2. QG: choi branch -> Question_Generation.ipynb -> train.md

### 과정
1. PororoMRC: inference과정에서 mrc reader 부분을 PororoMRC로 수정  
  1-1. retrieval에서 k가 1보다 클 경우 context를 합쳐주는 구문에서 ' '.join -> '♧'.join으로 수정
```python
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context_id": doc_indices[idx],
                    "context": "♧".join(  # 기존에는 ' '.join()
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
```
  1-2. inference만 수정한 후 baseline와 동일하게 진행  
2. QG: Question_Generation.ipynb에서 조건에 맞도록 Data 추가 -> train.py를 추가한 데이터를 함께 활용할 수 있도록 수정하여 활용.  
  2-1. Question_Generation.ipynb에서 csv형태로 augmentation data 저장  
  2-2. train만 수정한 후 baseline과 동일하게 진행  
  
### 결과
1. PororoMRC

|Experiments|Top-k|Epoch|Valid-EM|Valid-F1|Test-EM|Test-F1|
|---|---|---|---|---|---|---|
|Baseline|1|3|21.666|25.664|16.250|26.210|
|Baseline|50|3|36.666|42.649|39.170|50.310|
|PororoMRC|1|3|10.833|15.388|11.250|23.930|
|PororoMRC|50|3|18.333|25.210|No_Try|No_Try|

- 기존 Baseline에 비해 훨씬 성능이 떨어짐을 확인할 수 있었다.
- KorQuAD로 학습한 것은 Generality가 낮은 것 같다. 즉, 'KorQuAD로 학습 -> 내 데이터로 확인' 했을 때 성능이 많이 떨어지는 것을 확인할 수 있다.
- 그렇다면, klue/bert-base에 추가로 KorQuAD를 학습하여 진행하는 것은 성능의 개선으로 늘지 않을까?
- 혹여나, Bias가 높은 KorQuAD로 인해서 성능이 오히려 떨어질 수도 있다.

2. Question Generation

|Experiments|Top-k|Epoch|Valid-EM|Valid-F1|Test-EM|Test-F1|
|---|---|---|---|---|---|---|
|Baseline|20|2|37.5|46.101|42.920|54.653|
|QG-ALL|20|2|32.916|39.327|39.170|49.860|
|QG-SUB|20|2|35.0|43.473|42.920|54.690|

- QG-ALL
  - title을 포함하고 있는 모든 text에 대한 Question Generation Data를 활용
  - 약 28,000개의 Data 추가
- QG-SUB
  - title을 포함하고 있는 모든 text에 더하여 2가지 추가적 제약 추가
  - 생성된 질문에 title을 포함하고 있지 않아야함.
  - 생성된 질문과 text 사이의 cosine 유사도가 60점 이상일 경우에만 활용
  - 약 5,000개의 Data 추가  
- 결과
  - QA-ALL의 경우 Baseline에 비해 성능이 떨어지는 것을 볼 수 있다.
  - QA-SUB의 경우 Baseline과 거의 비슷한 성능을 내고 있기 때문에, 추가적으로 Augmentation을 해보는 것은 최종 결과를 낼 때만 한 번 시도를 해보면 좋을 것이라고 판단된다.
   

