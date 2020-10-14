#!/usr/bin/env bash

MAX_SEQ_LENGTH=256
MODEL_NAME_OR_PATH=bert-base-uncased
DATA_DIR=data
LABELS=${DATA_DIR}/labels.txt
CACHE_DIR=/home/qbbao/002_files/cache_transformers-3.0
OUTPUT_DIR=outputs/${MODEL_NAME_OR_PATH}_${MAX_SEQ_LENGTH}
NUM_TRAIN_EPOCH=5
PER_DEVICE_TRAIN_BATCH_SIZE=16
PER_DEVICE_EVAL_BATCH_SIZE=32
LOGGING_STEPS=500
SAVE_STEPS=500

python run_ner.py \
--do_train \
--do_eval \
--do_predict \
--evaluate_during_training \
--max_seq_length ${MAX_SEQ_LENGTH} \
--model_name_or_path ${MODEL_NAME_OR_PATH} \
--data_dir ${DATA_DIR} \
--labels ${LABELS} \
--cache_dir ${CACHE_DIR} \
--output_dir ${OUTPUT_DIR} \
--num_train_epochs ${NUM_TRAIN_EPOCH} \
--per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
--per_device_eval_batch_size ${PER_DEVICE_EVAL_BATCH_SIZE} \
--logging_steps ${LOGGING_STEPS} \
--save_steps ${SAVE_STEPS}
