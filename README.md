# Open-Domain Question Answering Competition



## Code

- Reader
  - README
  - model.py: LSTM Model, Normal Model
  - train.py: baseline training code
  - train_kfold.py: KFold LSTM training model
  - train_with_negatives.py: elastic search & training code
  - inference.py
  - inference_dpr.py
- Retrieval
  - README
  - Bi-Encoder In-batch Negative
  - Bi-Encoder with Efficient Negative
  - cross_encoder In-batch Negative
  - cross_encoder with Elastic Negative
  - elastic_search.py
  - colbert.py
- utils
  - loss.py
  - utils_qa.py
  
```
  │  .gitignore
  │  config.yaml
  │  inference.py
  │  inference_dpr.py
  │  install_requirements.sh
  │  README.md
  │  
  ├─reader
  │      inference_dpr_kfold.sh
  │      models.py
  │      README.md
  │      train.py
  │      trainer_qa.py
  │      train_kfold.py
  │      train_kfold.sh
  │
  ├─retrieval
  │      ColBert.py
  │      cross_encoder.py
  │      dense_train.py
  │      DPR.py
  │      elastic_filter_POS.py
  │      elastic_make_negatives.py
  │      elastic_search.py
  │      gold.py
  │      README.md
  │      retrieval.py
  │
  └─utils
          loss.py
          utils_qa.py
          __init__.py
          
```


## Implementation

- Install Requirements

  ```py
  bash install_requirements.sh
  ```

