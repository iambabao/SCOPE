# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2021/1/18
@Desc               :
@Last modified by   : Bao
@Last modified date : 2021/1/18
"""

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from src.models import BertClassifier
from src.utils import read_json_lines, save_json_lines


def main():
    data = list(read_json_lines('../data/baseline/ENT/test_outputs.json'))
    tokenizer = AutoTokenizer.from_pretrained('checkpoints/0118/bert_bert-base-cased_256/checkpoint-14000')
    model = BertClassifier.from_pretrained('checkpoints/0118/bert_bert-base-cased_256/checkpoint-14000').to('cuda')

    outputs = []
    for line in tqdm(data):
        context = line['context']
        golden = line['golden']
        predicted = []
        for phrase, start, end in line['predicted']:
            highlight = '{} <hl> {} <hl> {}'.format(context[:start], context[start:end], context[end:])
            encoded = tokenizer.encode_plus(
                highlight,
                phrase,
                padding="max_length",
                truncation="longest_first",
                max_length=512,
                return_tensors='pt',
            )
            for key, value in encoded.items():
                encoded[key] = value.to('cuda')
            logits = model(**encoded)[0]
            if np.argmax(logits.detach().cpu().numpy()) == 1:
                predicted.append((phrase, start, end))
        outputs.append({'context': context, 'golden': golden, 'predicted': predicted})
    print(len(outputs))

    save_json_lines(outputs, 'data/test_outputs.json')


if __name__ == '__main__':
    main()
