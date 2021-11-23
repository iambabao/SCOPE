#!/usr/bin/env bash

DATE=$(date +%m%d)
SEED=2333

# BERT
#MODEL_TYPE="bert"
#MODEL_NAME_OR_PATH="bert-base-cased"
#MODEL_TYPE="bert-gnn"
#MODEL_NAME_OR_PATH="bert-base-cased"

# SpanBERT
#MODEL_TYPE="bert"
#MODEL_NAME_OR_PATH="SpanBERT/spanbert-base-cased"
#MODEL_TYPE="bert-gnn"
#MODEL_NAME_OR_PATH="SpanBERT/spanbert-base-cased"

# RoBERTa
#MODEL_TYPE="roberta"
#MODEL_NAME_OR_PATH="roberta-base"
MODEL_TYPE="roberta-gnn"
MODEL_NAME_OR_PATH="roberta-base"

MAX_SEQ_LENGTH=256
LOSS_TYPE="pu"
PRIOR=2.00

DATA_TYPE="SQuAD"
#DATA_TYPE="DROP"
DATA_DIR="../data/${DATA_TYPE}"
OUTPUT_DIR="checkpoints/${DATA_TYPE}/${DATE}"
CACHE_DIR="${HOME}/003_downloads/cache_transformers"
LOG_DIR="log/${DATA_TYPE}/${DATE}"

NUM_TRAIN_EPOCH=15
LEARNING_RATE=2e-5
GRADIENT_ACCUMULATION_STEPS=1
PER_GPU_TRAIN_BATCH_SIZE=12
PER_GPU_EVAL_BATCH_SIZE=12
LOGGING_STEPS=1000
SAVE_STEPS=1000

CUDA_VISIBLE_DEVICES=0 python train_extractor.py \
--model_type ${MODEL_TYPE} \
--model_name_or_path ${MODEL_NAME_OR_PATH} \
--max_seq_length ${MAX_SEQ_LENGTH} \
--loss_type ${LOSS_TYPE} \
--prior ${PRIOR} \
--data_dir ${DATA_DIR} \
--output_dir ${OUTPUT_DIR} \
--cache_dir ${CACHE_DIR} \
--log_dir ${LOG_DIR} \
--do_train \
--save_best \
--overwrite_output_dir \
--num_train_epochs ${NUM_TRAIN_EPOCH} \
--learning_rate ${LEARNING_RATE} \
--gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
--per_gpu_train_batch_size ${PER_GPU_TRAIN_BATCH_SIZE} \
--do_eval \
--evaluate_during_training \
--per_gpu_eval_batch_size ${PER_GPU_EVAL_BATCH_SIZE} \
--logging_steps ${LOGGING_STEPS} \
--save_steps ${SAVE_STEPS} \
--seed ${SEED}
