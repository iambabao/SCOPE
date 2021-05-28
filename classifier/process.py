# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2021/1/18
@Desc               :
@Last modified by   : Bao
@Last modified date : 2021/1/18
"""

import logging
import stanza
from tqdm import tqdm

from src.utils import init_logger, read_json_lines, save_json_lines

logger = logging.getLogger(__name__)


def extract_ent(data_type, role):
    en_nlp = stanza.Pipeline('en', processors='tokenize,ner')

    data = []
    for line in tqdm(list(read_json_lines('../data/{}_ENT/{}_outputs.json'.format(data_type, role)))):
        predicted = []
        doc = en_nlp(line['context'])
        for sent in doc.sentences:
            for ent in sent.ents:
                predicted.append((ent.text, ent.start_char, ent.end_char))
        data.append({'context': line['context'], 'golden': line['golden'], 'predicted': predicted})
    save_json_lines(data, '../data/{}_END/{}_outputs.json'.format(data_type, role))


def convert_data(data_type, role):
    data = []
    for line in read_json_lines('../data/{}_ENT/{}_outputs.json'.format(data_type, role)):
        context = line['context']
        golden = [(v[1], v[2]) for v in line['golden']]
        predicted = [(v[1], v[2]) for v in line['predicted']]

        positive = [(context[v[0]:v[1]], v[0], v[1]) for v in predicted if v in golden]
        negative = [(context[v[0]:v[1]], v[0], v[1]) for v in predicted if v not in golden]
        phrases = positive + negative
        labels = [1] * len(positive) + [0] * len(negative)

        data.append({'context': context, 'phrases': phrases, 'labels': labels, 'golden': line['golden']})

    save_json_lines(data, '../data/{}_CLS/data_{}.json'.format(data_type, role))


def main():
    init_logger(logging.INFO)

    convert_data('SQuAD', 'train')
    convert_data('SQuAD', 'eval')
    convert_data('SQuAD', 'test')

    convert_data('DROP', 'train')
    convert_data('DROP', 'eval')
    convert_data('DROP', 'test')



if __name__ == '__main__':
    main()
