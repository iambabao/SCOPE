# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2020/10/13
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/11/12
"""

import logging
import argparse
import os
import torch
from tqdm import tqdm

from src.models import Pipeline
from src.utils import init_logger, log_title, save_json, read_json_lines

logger = logging.getLogger(__name__)


def read_data(filename, use_golden=True):
    data = []
    for line in tqdm(list(read_json_lines(filename)), desc='Reading data'):
        key = 'golden' if use_golden else 'predicted'
        for answer in line[key]:
            data.append({'context': line['context'], 'answer': answer})
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str, help="Checkpoint to evaluate")
    parser.add_argument("--beam_size", default=5, type=int, help="Number of questions to be generated")
    parser.add_argument(
        "--max_seq_length",
        default=None,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization",
    )
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU")
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    args = parser.parse_args()

    init_logger(logging.INFO)

    golden_data = read_data(
        '../phrase_extractor/checkpoints/{}/test_outputs.json'.format(args.checkpoint),
        use_golden=True,
    )
    predicated_data = read_data(
        '../phrase_extractor/checkpoints/{}/test_outputs.json'.format(args.checkpoint),
        use_golden=False,
    )
    pipeline = Pipeline(
        generator_model_name='valhalla/t5-base-qg-hl',
        reader_model_name='deepset/bert-base-cased-squad2',
        cache_dir=args.cache_dir,
        device='cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu',
    )
    os.makedirs('../data/qg_qa/{}'.format(args.checkpoint), exist_ok=True)

    logger.info(log_title('Golden'))
    golden_results = pipeline(golden_data, args.beam_size, args.max_seq_length, args.batch_size)
    save_json(golden_results, '../data/qg_qa/{}/golden_results.json'.format(args.checkpoint))

    logger.info(log_title('Predicted'))
    predicated_results = pipeline(predicated_data, args.beam_size, args.max_seq_length, args.batch_size)
    save_json(predicated_results, '../data/qg_qa/{}/predicated_results.json'.format(args.checkpoint))

    delta_results = [v for u, v in zip(predicated_data, predicated_results) if u not in golden_data]
    save_json(delta_results, '../data/qg_qa/{}/delta_results.json'.format(args.checkpoint))


if __name__ == '__main__':
    main()
