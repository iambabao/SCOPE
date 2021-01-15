# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/7/23
@Desc               : 
@Last modified by   : Bao
@Last modified date : 2021/1/14
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

        for span in entry['phrase_spans']:
            start, end = span[1], span[2]
            # skip nested data
            if not all([v == 'O' for v in tags[start:end + 1]]): continue
            tags[start] = 'B-Answer'
            for i in range(start + 1, end + 1): tags[i] = 'I-Answer'

        for token, tag in zip(tokens, tags):
            print('{} {}'.format(token, tag), file=fp)
        print('', file=fp)


def convert_data_nested(filein, fileout, max_num_nested):
    token2tag = []
    for entry in tqdm(list(read_json_lines(filein)), desc='Converting data'):
        tokens = []
        tags = []
        for spans in entry['token_spans']:
            for token, start, end in spans:
                tokens.append(token)
                tags.append(['O'])

        for phrase, start, end in entry['phrase_spans']:
            flag = True
            for i in range(len(tags[0])):
                if flag and all([v[i] == 'O' for v in tags[start:end + 1]]):
                    flag = False
                    tags[start][i] = 'B-Answer'
                    for j in range(start + 1, end + 1):
                        tags[j][i] = 'I-Answer'
            if flag:
                for i in range(len(tags)):
                    tags[i].append('O')
                tags[start][-1] = 'B-Answer'
                for j in range(start + 1, end + 1):
                    tags[j][-1] = 'I-Answer'
        token2tag.append((tokens, tags))

    logger.info('Maximum nested level: {}'.format(max([len(tags[0]) for tokens, tags in token2tag])))

    fp = open(fileout, 'w', encoding='utf-8')
    for tokens, tags in token2tag:
        for token, tag_list in zip(tokens, tags):
            tag_list = tag_list[:max_num_nested] + ['O'] * (max_num_nested - len(tag_list))
            print('{}\t{}'.format(token, '\t'.join(tag_list)), file=fp)
        print('', file=fp)


def main():
    init_logger(logging.INFO)

    convert_data('../data/phrase/data_train.feature.json', '../data/ner/data_train.txt')
    convert_data('../data/phrase/data_eval.feature.json', '../data/ner/data_eval.txt')
    convert_data('../data/phrase/data_test.feature.json', '../data/ner/data_test.txt')

    convert_data_nested('../data/phrase/data_train.feature.json', '../data/ner/data_train.nested.txt', max_num_nested=4)
    convert_data_nested('../data/phrase/data_eval.feature.json', '../data/ner/data_eval.nested.txt', max_num_nested=4)
    convert_data_nested('../data/phrase/data_test.feature.json', '../data/ner/data_test.nested.txt', max_num_nested=4)


if __name__ == '__main__':
    main()
