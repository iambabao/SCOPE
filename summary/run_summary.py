# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2021/1/5
@Desc               :
@Last modified by   : Bao
@Last modified date : 2021/1/11
"""

import logging
import argparse
import json
import requests
from tqdm import tqdm
from nltk import sent_tokenize

from utils import init_logger, read_json_lines, save_json_lines

logger = logging.getLogger(__name__)

url = 'http://10.176.64.113:22222/bert_sum'
headers = {'content-type': 'application/json'}


def run_summary(filein, fileout):
    data = []
    for entry in tqdm(list(read_json_lines(filein))):
        paragraph = entry['context']
        sentences = sent_tokenize(paragraph)
        sentence_spans = []
        for sentence in sentences:
            start = paragraph.index(sentence)
            sentence_spans.append((start, start + len(sentence)))

        response = requests.post(url, data=json.dumps({'sentences': sentences}), headers=headers)
        scores = response.json()['scores']

        for phrase, phrase_start, phrase_end in entry['predicted']:
            start, end = 0, 0
            for i, (s, e) in enumerate(sentence_spans):
                if s <= phrase_start < e: start = i
                if s < phrase_end <= e: end = i
            score_w = sum(scores[start:end + 1]) / (end - start + 1)
            data.append({'context': paragraph, 'answer': (phrase, phrase_start, phrase_end), 'score_w': score_w})

    save_json_lines(data, fileout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, type=str, help="The input file")
    parser.add_argument("--output_file", required=True, type=str, help="The output file")
    args = parser.parse_args()

    init_logger(logging.INFO)

    run_summary(args.input_file, args.output_file)


if __name__ == '__main__':
    main()
