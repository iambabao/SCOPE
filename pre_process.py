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
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from collections import defaultdict

from utils import init_logger, read_file, read_json, save_json, save_json_lines

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
        title = page['title']
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
                        'id': _id,
                        'title': title,
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
        title = page['title']
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
                        'id': _id,
                        'title': title,
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
    train_data = convert_data_with_nltk('data/SQuAD/train-v2.0.json', 'data/SQuAD/nltk_train_sentence.json')
    dev_data = convert_data_with_nltk('data/SQuAD/dev-v2.0.json', 'data/SQuAD/nltk_dev_sentence.json')

    """
    2020-10-16 15:12:52 - INFO - __main__:  match: 86460
    2020-10-16 15:12:52 - INFO - __main__:  mismatch: 361
    2020-10-16 15:15:21 - INFO - __main__:  match: 10360
    2020-10-16 15:15:21 - INFO - __main__:  mismatch: 28
    """
    # train_data = convert_data_with_stanza('data/SQuAD/train-v2.0.json', 'data/SQuAD/stanza_train_sentence.json')
    # dev_data = convert_data_with_stanza('data/SQuAD/dev-v2.0.json', 'data/SQuAD/stanza_dev_sentence.json')

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
        else:
            logger.info(line['title'])
    for line in dev_data:
        if line in dev_data:
            if line['title'] in eval_titles:
                eval_split.append(line)
            else:
                logger.info(line['title'])

    train_split = refine_data(train_split)
    eval_split = refine_data(eval_split)
    test_split = refine_data(test_split)
    save_json_lines(train_split, 'data/phrase/data_train.json')
    save_json_lines(eval_split, 'data/phrase/data_eval.json')
    save_json_lines(test_split, 'data/phrase/data_test.json')


if __name__ == '__main__':
    main()
