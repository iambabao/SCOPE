# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/7/23
@Desc               : 
@Last modified by   : Bao
@Last modified date : 2020/7/23
"""

import logging
from tqdm import tqdm
from collections import defaultdict

from utils import init_logger, read_file

logger = logging.getLogger(__name__)


def generate_data(input_file, output_file):
    data = defaultdict(list)
    for line in tqdm(list(read_file(input_file)), desc='Loading data: {}'.format(input_file)):
        sequence, start_end, _, _, _, _, _, _, _, _ = line.strip().split('\t')
        ans_start, ans_end = map(int, start_end.split())
        data[sequence].append((ans_start, ans_end))

    fp = open(output_file, 'w', encoding='utf-8')
    for sequence, positions in tqdm(data.items(), desc='Converting data'):
        sequence = sequence.split()
        tag = ['O'] * len(sequence)
        for ans_start, ans_end in positions:
            if not all([v == 'O' for v in tag[ans_start:ans_end + 1]]): continue
            tag[ans_start] = 'B-Answer'
            for i in range(ans_start + 1, ans_end + 1): tag[i] = 'I-Answer'
        for w, t in zip(sequence, tag):
            print('{} {}'.format(w, t), file=fp)
        print('', file=fp)


def main():
    logger.setLevel(logging.INFO)
    init_logger(logging.INFO)

    logger.info('generating data...')
    generate_data('../data/redistribute/raw/train.txt', 'data/train.txt')
    generate_data('../data/redistribute/raw/dev.txt.shuffle.dev', 'data/dev.txt')
    generate_data('../data/redistribute/raw/dev.txt.shuffle.test', 'data/test.txt')


if __name__ == '__main__':
    main()
