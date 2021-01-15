# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2020/10/13
@Desc               :
@Last modified by   : Bao
@Last modified date : 2021/1/13
"""

import argparse
import logging
import torch

from src.models import Pipeline
from src.utils import init_logger, save_json, read_json_lines

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, type=str, help="The input file")
    parser.add_argument("--output_file", required=True, type=str, help="The output file")
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

    logger.info('Initializing pipeline')
    pipeline = Pipeline(
        generator_model_name='valhalla/t5-base-qg-hl',
        reader_model_name='deepset/bert-base-cased-squad2',
        cache_dir=args.cache_dir,
        device='cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu',
    )

    logger.info('Running pipeline')
    data = list(read_json_lines(args.input_file))
    results = pipeline(data, args.beam_size, args.max_seq_length, args.batch_size)
    save_json(results, args.output_file)


if __name__ == '__main__':
    main()
