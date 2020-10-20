# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2020/10/13
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/10/13
"""

import logging
import os
import torch
from tqdm import tqdm

from src.models import Pipeline
from src.utils import init_logger, log_title, read_json, save_json, read_json_lines

logger = logging.getLogger(__name__)


def read_data(filename, use_golden=True):
    data = []
    for line in tqdm(list(read_json_lines(filename)), desc='Reading data'):
        context = line['context']
        key = 'golden' if use_golden else 'predicted'
        for answer in line[key]:
            data.append({
                'context': context.replace(answer, '<hl> {} <hl>'.format(answer)),
                'raw_answer': answer
            })
    return data


def run(checkpoint):
    golden_data = read_data(
        '../phrase_extractor/checkpoints/{}/eval_outputs.json'.format(checkpoint),
        use_golden=True,
    )
    predicated_data = read_data(
        '../phrase_extractor/checkpoints/{}/eval_outputs.json'.format(checkpoint),
        use_golden=False,
    )
    pipeline = Pipeline(
        generator_model_name='valhalla/t5-base-qg-hl',
        reader_model_name='deepset/bert-base-cased-squad2',
        lm_model_name='gpt2',
        cache_dir='/home/qbbao/003_downloads/cache_transformers-3.1.0',
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    os.makedirs('../data/qg_qa/{}'.format(checkpoint), exist_ok=True)

    logger.info(log_title('Golden'))
    batch_size = 10
    num_batches = (len(golden_data) + batch_size - 1) // batch_size
    golden_results = []
    for i in tqdm(range(num_batches)):
        start = i * batch_size
        end = min(len(golden_data), (i + 1) * batch_size)
        batch_data = golden_data[start:end]
        batch_results = pipeline(batch_data, lm_key='question')
        golden_results.extend(batch_results)
    save_json(golden_results, '../data/qg_qa/{}/golden_results.json'.format(checkpoint))

    logger.info(log_title('Predicted'))
    batch_size = 8
    num_batches = (len(predicated_data) + batch_size - 1) // batch_size
    predicated_results = []
    for i in tqdm(range(num_batches)):
        start = i * batch_size
        end = min(len(predicated_data), (i + 1) * batch_size)
        batch_data = predicated_data[start:end]
        batch_results = pipeline(batch_data, lm_key='question')
        predicated_results.extend(batch_results)
    save_json(predicated_results, '../data/qg_qa/{}/predicated_results.json'.format(checkpoint))

    delta_results = [v for v in predicated_results if v not in golden_results]
    save_json(delta_results, '../data/qg_qa/{}/delta_results.json'.format(checkpoint))


def main():
    init_logger(logging.INFO)

    run('1018/bert_bert-base-cased_128_masked_pu-loss_0.20')


if __name__ == '__main__':
    main()
