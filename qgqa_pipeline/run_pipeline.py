# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2020/10/13
@Desc               :
@Last modified by   : Bao
@Last modified date : 2021/5/11
"""

import argparse
import logging
import torch

from src.models import Pipeline
from src.utils import init_logger, read_json_lines, save_json_lines

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator_model", required=True, type=str, help="The path to the generator model")
    parser.add_argument("--reader_model", required=True, type=str, help="The path to the reader model")
    parser.add_argument("--input_file", required=True, type=str, help="The input file")
    parser.add_argument("--output_file", required=True, type=str, help="The output file")
    parser.add_argument(
        "--max_seq_length",
        default=None,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization",
    )
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU")
    parser.add_argument("--beam_size", default=5, type=int, help="Number of questions to be generated")
    parser.add_argument("--temperature", default=1.0, type=float, help="Temperature in softmax")
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    args = parser.parse_args()

    init_logger(logging.INFO)

    logger.info('Loading data')
    input_data = list(read_json_lines(args.input_file))

    logger.info('Initializing pipeline')
    pipeline = Pipeline(
        generator_model=args.generator_model,
        reader_model=args.reader_model,
        cache_dir=args.cache_dir,
        device='cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu',
    )

    logger.info('Running pipeline')
    outputs = pipeline(input_data, args.max_seq_length, args.batch_size, args.beam_size, args.temperature)

    save_json_lines(outputs, args.output_file)


if __name__ == '__main__':
    main()
