# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2021/1/5
@Desc               :
@Last modified by   : Bao
@Last modified date : 2021/1/11
"""

import os
import json
import logging
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from gensim.summarization import keywords

from utils import init_logger, read_json, save_json, read_json_lines

logger = logging.getLogger(__name__)


def summ_with_textrank(filename):
    score_file = '../data/summary/text_rank.json'
    if os.path.exists(score_file):
        num2score = read_json(score_file)
    else:
        num2score = defaultdict(list)
        for entry in tqdm(list(read_json_lines(filename))):
            paragraph = entry['context']
            key_words = keywords(paragraph, ratio=1.0, scores=True)

            for tokens in entry['token_spans']:
                start_index = tokens[0][1]
                end_index = tokens[-1][2]
                sentence = entry['context'][start_index:end_index]

                num_spans = min(6, len([v for v, _, _ in entry['phrase_spans'] if v in sentence]))
                scores = [score for word, score in key_words if word in sentence]
                if len(scores) == 0:
                    num2score[num_spans].append(0)
                else:
                    num2score[num_spans].append(sum(scores))
        num2score = {key: sum(value) / len(value) for key, value in sorted(num2score.items())}
        save_json(num2score, score_file)
    logger.info(json.dumps(num2score, ensure_ascii=False, indent=4))

    return num2score


def summ_with_bertsum(filename):
    score_file = '../data/summary/bert_sum.json'
    if os.path.exists(score_file):
        num2score = read_json(score_file)
    else:
        url = 'http://10.176.64.113:22222/bert_sum'
        headers = {'content-type': 'application/json'}
        num2score = defaultdict(list)
        for entry in tqdm(list(read_json_lines(filename))):
            paragraph = entry['context']
            sentences = [paragraph[spans[0][1]:spans[-1][2]] for spans in entry['token_spans']]

            response = requests.post(url, data=json.dumps({'sentences': sentences}), headers=headers)
            scores = response.json()['scores']

            for sentence, score in zip(sentences, scores):
                num_spans = min(6, len([v for v, _, _ in entry['phrase_spans'] if v in sentence]))
                num2score[num_spans].append(score)
        num2score = {key: sum(value) / len(value) for key, value in sorted(num2score.items())}
        save_json(num2score, score_file)
    logger.info(json.dumps(num2score, ensure_ascii=False, indent=4))

    return num2score


def main():
    init_logger(logging.INFO)

    labels = ['0', '1', '2', '3', '4', '5', '>5']
    textrank_scores = summ_with_textrank('../data/phrase/data_train.feature.json')
    bertsum_scores = summ_with_bertsum('../data/phrase/data_train.feature.json')

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    axs[0].stackplot(labels, textrank_scores.values(), linewidth=2, color='aquamarine', alpha=0.3)
    axs[0].set_xlim([0, 7])
    axs[0].set_ylim([0, 1.5])
    axs[0].set_title('TextRank', fontsize=20)
    axs[0].set_xlabel('number of spans', fontsize=15)
    axs[0].set_ylabel('sentence salience', fontsize=15)
    axs[0].spines['top'].set_color('none')
    axs[0].spines['right'].set_color('none')

    axs[1].stackplot(labels, bertsum_scores.values(), linewidth=2, color='aquamarine', alpha=0.3)
    axs[1].set_xlim([0, 7])
    axs[1].set_ylim([0, 0.5])
    axs[1].set_title('BERTSum', fontsize=20)
    axs[1].set_xlabel('number of spans', fontsize=15)
    axs[1].set_ylabel('sentence salience', fontsize=15)
    axs[1].spines['top'].set_color('none')
    axs[1].spines['right'].set_color('none')

    fig.tight_layout()
    plt.savefig('../data/summary/fig_score.pdf')
    # plt.show()


if __name__ == '__main__':
    main()
