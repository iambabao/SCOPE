#!/usr/bin/env bash

MAX_SEQ_LENGTH=512
BATCH_SIZE=16
BEAM_SIZE=5
TEMPERATURE=1.5
CACHE_DIR="${HOME}/003_downloads/cache_transformers"

GENERATOR_MODEL="valhalla/t5-base-qg-hl"
READER_MODEL="deepset/roberta-base-squad2"
INPUT_FILE="../data/temp/input.json"
OUTPUT_FILE="../data/temp/output.json"

CUDA_VISIBLE_DEVICES=0 python run_pipeline.py \
--generator_model ${GENERATOR_MODEL} \
--reader_model ${READER_MODEL} \
--input_file ${INPUT_FILE} \
--output_file ${OUTPUT_FILE} \
--max_seq_length ${MAX_SEQ_LENGTH} \
--batch_size ${BATCH_SIZE} \
--beam_size ${BEAM_SIZE} \
--temperature ${TEMPERATURE} \
--cache_dir ${CACHE_DIR}
