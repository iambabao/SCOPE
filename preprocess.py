# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/6/16
@Desc               : 
@Last modified by   : Bao
@Last modified date : 2020/10/19
"""

import logging
import stanza
import random
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from collections import defaultdict

from utils import init_logger, read_json, save_json, save_json_lines

logger = logging.getLogger(__name__)


def locate_answer_with_nltk(context, sentences, answer, raw_start):
    for sentence in sentences:
        sentence_start = context.index(sentence)
        if sentence_start <= raw_start < raw_start + len(answer) <= sentence_start + len(sentence):
            new_start = raw_start - sentence_start
            if sentence[new_start:new_start + len(answer)] == answer:
                return sentence, new_start, new_start + len(answer)
    return '', 0, 0


def convert_data_with_nltk(filein, fileout):
    data = read_json(filein)['data']

    mismatch = 0
    new_data = []
    for page in tqdm(data, desc='Converting data: {}'.format(filein)):
        for paragraph in page['paragraphs']:
            context = paragraph['context']
            sentences = sent_tokenize(context)
            for qa in paragraph['qas']:
                if qa['is_impossible']:
                    continue
                _id = qa['id']
                question = qa['question']
                answers = []
                for answer in qa['answers']:
                    if answer in answers:
                        continue
                    answers.append(answer)
                    sentence, answer_start, answer_end = locate_answer_with_nltk(
                        context, sentences, answer['text'], answer['answer_start']
                    )
                    if len(sentence) == 0 or answer_start >= answer_end:
                        mismatch += 1
                        logger.warning(_id)
                        continue
                    new_data.append({
                        'context': sentence,
                        'question': question,
                        'answer': answer['text'],
                        'answer_start': answer_start,
                        'answer_end': answer_end,
                    })
    logger.info('match: {}'.format(len(new_data)))
    logger.info('mismatch: {}'.format(mismatch))
    save_json(new_data, fileout)
    return new_data


def locate_answer_with_stanza(doc, answer, raw_start):
    for sentence in doc.sentences:
        first_token = sentence.tokens[0]
        last_token = sentence.tokens[-1]
        if first_token.start_char <= raw_start < raw_start + len(answer) <= last_token.end_char:
            new_start = raw_start - first_token.start_char
            if sentence.text[new_start:new_start + len(answer)] == answer:
                return sentence.text, new_start, new_start + len(answer)
    return '', 0, 0


def convert_data_with_stanza(filein, fileout):
    nlp = stanza.Pipeline('en')
    data = read_json(filein)['data']

    mismatch = 0
    new_data = []
    for page in tqdm(data, desc='Converting data: {}'.format(filein)):
        for paragraph in page['paragraphs']:
            context = paragraph['context']
            doc = nlp(context)
            for qa in paragraph['qas']:
                if qa['is_impossible']:
                    continue
                _id = qa['id']
                question = qa['question']
                answers = []
                for answer in qa['answers']:
                    if answer in answers:
                        continue
                    answers.append(answer)
                    sentence, answer_start, answer_end = locate_answer_with_stanza(
                        doc, answer['text'], answer['answer_start']
                    )
                    if len(sentence) == 0 or answer_start >= answer_end:
                        mismatch += 1
                        logger.warning(_id)
                        continue
                    new_data.append({
                        'context': sentence,
                        'question': question,
                        'answer': answer['text'],
                        'answer_start': answer_start,
                        'answer_end': answer_end,
                    })
    logger.info('match: {}'.format(len(new_data)))
    logger.info('mismatch: {}'.format(mismatch))
    save_json(new_data, fileout)
    return new_data


def refine_data(data):
    context2qas = defaultdict(list)

    for line in data:
        context2qas[line['context']].append({
            'question': line['question'],
            'answer': line['answer'],
            'answer_start': line['answer_start'],
            'answer_end': line['answer_end'],
        })

    data = [{'context': key, 'qas': value} for key, value in context2qas.items()]
    return data


def main():
    init_logger(logging.INFO)

    """
    2020-10-16 15:43:53 - INFO - __main__:  match: 86607
    2020-10-16 15:43:53 - INFO - __main__:  mismatch: 214
    2020-10-16 15:43:54 - INFO - __main__:  match: 10368
    2020-10-16 15:43:54 - INFO - __main__:  mismatch: 20
    """
    train_data = convert_data_with_nltk('data/SQuAD/train-v2.0.json', 'data/phrase/raw_nltk_train.json')
    dev_data = convert_data_with_nltk('data/SQuAD/dev-v2.0.json', 'data/phrase/raw_nltk_dev.json')

    """
    2020-10-16 15:12:52 - INFO - __main__:  match: 86460
    2020-10-16 15:12:52 - INFO - __main__:  mismatch: 361
    2020-10-16 15:15:21 - INFO - __main__:  match: 10360
    2020-10-16 15:15:21 - INFO - __main__:  mismatch: 28
    """
    # train_data = convert_data_with_stanza('data/SQuAD/train-v2.0.json', 'data/phrase/raw_stanza_train.json')
    # dev_data = convert_data_with_stanza('data/SQuAD/dev-v2.0.json', 'data/phrase/raw_stanza_dev.json')

    # split into train, eval and test set
    train_data = refine_data(train_data)
    dev_data = refine_data(dev_data)
    random.shuffle(dev_data)
    eval_data, test_data = dev_data[:len(dev_data) // 2], dev_data[len(dev_data) // 2:]
    save_json_lines(train_data, 'data/phrase/data_train.json')
    save_json_lines(eval_data, 'data/phrase/data_eval.json')
    save_json_lines(test_data, 'data/phrase/data_test.json')


if __name__ == '__main__':
    main()
