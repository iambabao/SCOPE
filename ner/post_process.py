# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2020/12/8
@Desc               :
@Last modified by   : Bao
@Last modified date : 2021/1/14
"""

import argparse
import logging
from tqdm import tqdm

from utils import init_logger, read_file, read_json_lines, save_json_lines

logger = logging.getLogger(__name__)


def convert_outputs(predicted_file, golden_file, output_file):
    def _generate_result(_tokens, _labels):
        _predicted = []
        _token_start = 0
        while _token_start < len(_labels):
            if _labels[_token_start] == 'B-Answer':
                _token_end = _token_start + 1
                while _token_end < len(_labels) and _labels[_token_end] == 'I-Answer':
                    _token_end += 1
                _predicted.append([_token_start, _token_end])
                _token_start = _token_end
            else:
                _token_start += 1
        return _predicted

    tokens = []
    labels = []
    predicted_results = []
    for line in tqdm(read_file(predicted_file), desc='Converting predicted results'):
        if len(line.strip()) == 0:
            predicted_results.append(_generate_result(tokens, labels))
            tokens = []
            labels = []
        else:
            token, label = line.strip().split()
            tokens.append(token)
            labels.append(label)

    outputs = []
    for i, line in tqdm(enumerate(read_json_lines(golden_file)), desc='Generating results'):
        context = line['context']
        token_spans = []
        for spans in line['token_spans']: token_spans.extend(spans)

        golden = []
        for phrase, start, end in line['phrase_spans']:
            golden.append((phrase, token_spans[start][1], token_spans[end][2]))
        predicted = []
        for start, end in predicted_results[i]:
            phrase = context[token_spans[start][1]:token_spans[end - 1][2]]
            predicted.append((phrase, token_spans[start][1], token_spans[end - 1][2]))

        outputs.append({'context': context, 'golden': golden, 'predicted': predicted})

    save_json_lines(outputs, output_file)
    return outputs


def compute_metrics(outputs):
    results = {'Golden': 0, 'Predicted': 0, 'Matched': 0}
    for item in outputs:
        golden = set([(v[1], v[2]) for v in item['golden']])
        predicted = set([(v[1], v[2]) for v in item['predicted']])
        results['Golden'] += len(golden)
        results['Predicted'] += len(predicted)
        results['Matched'] += len(golden & predicted)
    if results['Golden'] == 0:
        if results['Predicted'] == 0:
            results['Precision'] = results['Recall'] = results['F1'] = 1.0
        else:
            results['Precision'] = results['Recall'] = results['F1'] = 0.0
    else:
        if results['Matched'] == 0 or results['Predicted'] == 0:
            results['Precision'] = results['Recall'] = results['F1'] = 0.0
        else:
            results['Precision'] = results['Matched'] / results['Predicted']
            results['Recall'] = results['Matched'] / results['Golden']
            results['F1'] = 2 * results['Precision'] * results['Recall'] / (results['Precision'] + results['Recall'])
    results['average'] = sum([len(item['predicted']) for item in outputs]) / len(outputs)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predicted_file", type=str, required=True, help="The predicted file")
    parser.add_argument("--golden_file", type=str, required=True, help="The golden file")
    parser.add_argument("--output_file", type=str, required=True, help="The output file")
    args = parser.parse_args()

    init_logger(logging.INFO)

    outputs = convert_outputs(args.predicted_file, args.golden_file, args.output_file)
    results = compute_metrics(outputs)
    for key, value in results.items():
        logger.info('{}: {}'.format(key, value))


if __name__ == '__main__':
    main()
