# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2020/12/8
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/12/8
"""

import logging
import argparse
import os
import math
from tqdm import tqdm
from collections import Counter
from nltk.tokenize import word_tokenize

from src.utils import init_logger, read_json, save_json, read_json_lines

logger = logging.getLogger(__name__)


def calc_importance(ps, pd, pk):
    redundancy = 0
    for word in ps:
        if ps[word] == 0: continue
        redundancy += ps[word] * math.log(ps[word])

    relevance = 0
    for word in pd:
        if pd[word] == 0: continue
        relevance += ps[word] * math.log(pd[word])

    informativeness = 0
    for word in pk:
        if pk[word] == 0: continue
        informativeness -= ps[word] * math.log(pk[word])

    importance = -redundancy + 2 * relevance + informativeness

    return redundancy, relevance, informativeness, importance


def process(context, spans, pk):
    pd = Counter([word.lower() for word in word_tokenize(context) if word in pk])
    norm = sum(pd.values())
    for word in pd: pd[word] /= norm

    ps = Counter()
    for span in spans:
        ps.update([word.lower() for word in word_tokenize(span) if word in pk])
    norm = sum(ps.values())
    for word in ps: ps[word] /= norm

    return calc_importance(ps, pd, pk)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="The input file")
    parser.add_argument("--is_golden", action="store_true", help="Whether to load spans from golden qa pairs")
    args = parser.parse_args()

    init_logger(logging.INFO)

    pk = Counter(read_json('../data/pk.json'))

    all_red, all_rel, all_inf, all_imp, counter = 0, 0, 0, 0, 0
    for entry in tqdm(list(read_json_lines(args.input_file))):
        context = entry['context']
        if args.is_golden:
            spans = [qa['answer'] for qa in entry['qas']]
        else:
            spans = [v[0] for v in entry['predicted']]
        red, rel, inf, imp = process(context, spans, pk)
        all_red += red
        all_rel += rel
        all_inf += inf
        all_imp += imp
        counter += 1
    results = {
        'Red': all_red / counter,
        'Rel': all_rel / counter,
        'Inf': all_inf / counter,
        'Imp': all_imp / counter,
    }
    for key in results.keys():
        logger.info('{}: {}'.format(key, results[key]))
    prefix, filename = os.path.split(args.input_file)
    filename, _ = os.path.splitext(filename)
    output_file = os.path.join(prefix, '{}.importance.json'.format(filename))
    save_json(results, output_file)


if __name__ == '__main__':
    main()
