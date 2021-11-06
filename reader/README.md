# READER Code

## Inference 명령어
- output_dir : prediction 결과를 저장할 경로
- dataset_name : predict할 dataset 경로
- model_name_or_path : inference에 사용할 reader model 경로 or name
- overwrite_output_dir : 기존 output_dir에 있는 결과물 덮어쓸것인지
- per_device_eval_batch_size : inference에 넣을 dataset의 batch_size
- num_fold : KFold 사용한 inference 진행시 몇 Fold로 진행했는지 넣어줄 인자 (default 는 1로써 KFold가 진행하지 않았을 경우임)
