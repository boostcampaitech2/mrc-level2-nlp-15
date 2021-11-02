#!/bin/bash

python train_kfold.py --output_dir ./models/lstm_512 \
--per_device_train_batch_size 10 \
--max_seq_length 512 --evaluation_strategy steps \
--num_train_epochs 3 \
--eval_step 100 \
--metric_for_best_model exact_match \
--save_total_limit 5 \
--load_best_model_at_end --learning_rate 2e-5 \
--save_steps 100 --overwrite_output_dir \
--num_fold 3
