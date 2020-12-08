# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2020/12/8
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/12/8
"""

import argparse
import logging
from tqdm import tqdm

from utils import init_logger, read_file, save_json_lines

logger = logging.getLogger(__name__)


def convert_results(filein, fileout):
    def _generate_result(_tokens, _labels):
        _context = ' '.join(_tokens)
        _predicted = []
        _token_start = 0
        while _token_start < len(_labels):
            if _labels[_token_start] == 'B-Answer':
                _token_end = _token_start + 1
                while _token_end < len(_labels) and _labels[_token_end] == 'I-Answer':
                    _token_end += 1
                _span = ' '.join(_tokens[_token_start:_token_end])
                _char_start = len(' '.join(_tokens[:_token_start]))
                if _token_start != 0: _char_start += 1  # 1 for blank
                _char_end = len(' '.join(_tokens[:_token_end]))
                assert _span == _context[_char_start:_char_end]
                _predicted.append((_span, _char_start, _char_end))
                _token_start = _token_end
            else:
                _token_start += 1
        return {'context': _context, 'predicted': _predicted}

    tokens = []
    labels = []
    results = []
    for line in tqdm(read_file(filein), desc='Converting results'):
        if len(line.strip()) == 0:
            results.append(_generate_result(tokens, labels))
            tokens = []
            labels = []
        else:
            token, label = line.strip().split()
            tokens.append(token)
            labels.append(label)

    save_json_lines(results, fileout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="The input file")
    parser.add_argument("--output_file", type=str, required=True, help="The output file")
    args = parser.parse_args()

    init_logger(logging.INFO)

    logger.info('Converting {} to {}'.format(args.input_file, args.output_file))
    convert_results(args.input_file, args.output_file)


if __name__ == '__main__':
    main()
