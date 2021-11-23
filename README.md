# Harvesting More Answer Spans from Paragraph beyond Annotation

Source code for [Harvesting More Answer Spans from Paragraph beyond Annotation]()

## Requirements

- pytorch==1.8.0
- transformers==4.5.1
- pytorch-geometric

## Extractor

```bash
#!/usr/bin/env bash

DATE=$(date +%m%d)
SEED=2333

MODEL_TYPE="roberta-gnn"
MODEL_NAME_OR_PATH="roberta-base"

MAX_SEQ_LENGTH=256
LOSS_TYPE="pu"
PRIOR=2.00

DATA_TYPE="SQuAD"
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
```

### Running with different models
Use `MODEL_TYPE` and `MODEL_NAME_OR_PATH` to control the backbone and weather to ues GNN.
Candidate model list is provided in `do_train.sh`.

### Running with different loss types
The model use `crocess entropy loss` when `LOSS_TYPE="ce"` and use `pu loss` when `LOSS_TYPE="pu"`.
When training with `pu loss`, `PRIOR` controls the prior distribution scale which is `2.00` for `SQuAD` and `2.75` for DROP in our experiments.

## Scorer

### $ score_{q} $

$ score_{q} $ is implemented based on question generation and question answering pipeline.

### $ score_{w} $

$ score_{w} $ is implemented based on [Text Summarization with Pretrained Encoders](https://aclanthology.org/D19-1387.pdf).
We use the [code](https://github.com/nlpyang/PreSumm) released by the original paper.
