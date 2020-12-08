# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2020/11/24
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/12/8
"""

import logging
import argparse
import torch

from src.models import Extractor
from src.utils import init_logger, read_json_lines, save_json_lines

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to pretrained model")
    parser.add_argument("--input_file", type=str, required=True, help="The input file")
    parser.add_argument("--output_file", type=str, required=True, help="The output file")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU/CPU")
    parser.add_argument("--log_file", type=str, default=None, help="The log file")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    args = parser.parse_args()

    init_logger(logging.INFO , args.log_file)
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    logger.info('Initializing model from {}'.format(args.model_name_or_path))
    extractor = Extractor(args.model_name_or_path, device)

    input_data = list(read_json_lines(args.input_file))
    results = extractor(input_data, batch_size=args.batch_size)
    save_json_lines(results, args.output_file)


if __name__ == '__main__':
    main()
