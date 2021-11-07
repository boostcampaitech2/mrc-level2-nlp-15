#!/bin/bash

python inference_.py --output_dir ./models/lstm_512 \
--output_dir ./outputs/test_dataset/ \
--dataset_name ../data/test_dataset/ \
--model_name_or_path ./models/lstm_512/ \
--do_predict --overwrite_output_dir \
--metric_for_best_model exact_match \
--per_device_eval_batch_size 512 \
--num_fold 10
