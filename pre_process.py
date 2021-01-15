# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/6/16
@Desc               : 
@Last modified by   : Bao
@Last modified date : 2021/1/13
"""

import logging
import os
from tqdm import tqdm

from utils import init_logger, read_file, read_json, save_json_lines

logger = logging.getLogger(__name__)


def convert_data(filein):
    raw_data = read_json(filein)['data']

    new_data = []
    for page in tqdm(raw_data, desc='Converting data: {}'.format(filein)):
        title = page['title']
        for paragraph in page['paragraphs']:
            context = paragraph['context']
            phrase_spans = []
            for qa in paragraph['qas']:
                for answer in qa['answers']:
                    answer_span = (answer['text'], answer['answer_start'], answer['answer_start'] + len(answer['text']))
                    if answer_span not in phrase_spans:
                        phrase_spans.append(answer_span)
            new_data.append({'title': title, 'context': context, 'phrase_spans': phrase_spans})

    return new_data


def main():
    init_logger(logging.INFO)

    train_data = convert_data('data/SQuAD/train-v1.1.json')
    dev_data = convert_data('data/SQuAD/dev-v1.1.json')

    # split into train, eval and test set
    # refer to: https://github.com/xinyadu/nqg
    train_split = []
    eval_split = []
    test_split = []
    train_titles = [line.strip() for line in read_file('data/SQuAD/doclist-train.txt')]
    eval_titles = [line.strip() for line in read_file('data/SQuAD/doclist-eval.txt')]
    test_titles = [line.strip() for line in read_file('data/SQuAD/doclist-test.txt')]
    for line in train_data:
        if line['title'] in train_titles:
            train_split.append(line)
        elif line['title'] in test_titles:
            test_split.append(line)
    for line in dev_data:
        if line['title'] in eval_titles:
            eval_split.append(line)

    os.makedirs('data/phrase', exist_ok=True)
    save_json_lines(train_split, 'data/phrase/data_train.json')
    save_json_lines(eval_split, 'data/phrase/data_eval.json')
    save_json_lines(test_split, 'data/phrase/data_test.json')


if __name__ == '__main__':
    main()
