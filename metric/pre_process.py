# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2020/12/8
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/12/8
"""

import logging
import nltk
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from collections import Counter

from src.utils import init_logger, save_json_dict, read_json_lines

logger = logging.getLogger(__name__)


def generate_pk(data_file, pk_file, min_word_freq=10):
    counter = Counter()
    for entry in tqdm(list(read_json_lines(data_file)), desc='Reading data'):
        counter.update([word.lower() for word in word_tokenize(entry['context'])])
    logger.info('Number of words: {}'.format(len(counter)))

    word2freq = {}
    stopwords = nltk.corpus.stopwords.words('english')
    for word, freq in counter.most_common():
        if freq < min_word_freq: break
        if word in stopwords: continue
        word2freq[word] = freq
    logger.info('Number of frequent words: {}'.format(len(word2freq)))

    norm = sum(word2freq.values())
    pk = {key: value / norm for key, value in word2freq.items()}
    save_json_dict(pk, pk_file)


def main():
    init_logger(logging.INFO)

    generate_pk('../data/phrase/data_train.json', '../data/pk.json')


if __name__ == '__main__':
    main()
