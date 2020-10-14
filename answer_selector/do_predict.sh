#!/usr/bin/env bash

MAX_SEQ_LENGTH=256
MODEL_NAME_OR_PATH=bert-base-uncased
DATA_DIR=data
LABELS=${DATA_DIR}/labels.txt
CACHE_DIR=/home/qbbao/002_files/cache_transformers-3.0
OUTPUT_DIR=outputs/${MODEL_NAME_OR_PATH}_${MAX_SEQ_LENGTH}
PER_DEVICE_EVAL_BATCH_SIZE=32

python run_ner.py \
--do_predict \
--max_seq_length ${MAX_SEQ_LENGTH} \
--model_name_or_path ${OUTPUT_DIR} \
--data_dir ${DATA_DIR} \
--labels ${LABELS} \
--cache_dir ${CACHE_DIR} \
--output_dir ${OUTPUT_DIR} \
--per_device_eval_batch_size ${PER_DEVICE_EVAL_BATCH_SIZE}
