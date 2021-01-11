# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/7/23
@Desc               : 
@Last modified by   : Bao
@Last modified date : 2021/1/10
"""

import logging
from tqdm import tqdm

from utils import init_logger, read_json_lines

logger = logging.getLogger(__name__)


def convert_data(filein, fileout):
    fp = open(fileout, 'w', encoding='utf-8')
    for entry in tqdm(list(read_json_lines(filein)), desc='Converting data'):
        tokens = []
        for spans in entry['token_spans']:
            tokens.extend([v[0] for v in spans])
        tags = ['O'] * len(tokens)

        for span in sorted(entry['phrase_spans'], key=lambda x: (x[2] - x[1], x[1])):
            start, end = span[1], span[2]
            # skip overlapping
            if not all([v == 'O' for v in tags[start:end + 1]]): continue
            # annotate span
            tags[start] = 'B-Answer'
            for i in range(start + 1, end + 1): tags[i] = 'I-Answer'

        for token, tag in zip(tokens, tags):
            print('{} {}'.format(token, tag), file=fp)
        print('', file=fp)


def main():
    init_logger(logging.INFO)

    convert_data('../data/phrase/data_train.tree.json', '../data/ner/data_train.txt')
    convert_data('../data/phrase/data_eval.tree.json', '../data/ner/data_eval.txt')
    convert_data('../data/phrase/data_test.tree.json', '../data/ner/data_test.txt')


if __name__ == '__main__':
    main()
