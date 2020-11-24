# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2020/11/24
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/11/24
"""

import os
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from .bert import BertExtractor
from .roberta import RobertaExtractor

MODEL_MAPPING = {
    'bert': BertExtractor,
    'roberta': RobertaExtractor,
}


def generate_outputs(offset_mappings, start_predicted, end_predicted, phrase_predicted):
    outputs = []
    for mapping, start_flags, end_flags, phrase_flags in zip(
        offset_mappings, start_predicted, end_predicted, phrase_predicted
    ):
        predicted = []
        for i, is_start in enumerate(start_flags):
            if is_start != 1: continue
            for j, is_end in enumerate(end_flags):
                if is_end != 1: continue
                if phrase_flags[i][j] == 1:
                    start, end = mapping[i][0], mapping[j][1]
                    predicted.append((int(start), int(end)))
        outputs.append(predicted)
    return outputs


class Extractor:
    def __init__(self, model_name_or_path, device="cpu"):
        model_params = os.path.split(model_name_or_path)[-1]
        model_type, _, max_seq_length, _, _, _ = model_params.split('_')

        self.model_type = model_type
        self.max_seq_length = int(max_seq_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.model = MODEL_MAPPING[model_type].from_pretrained(model_name_or_path)
        self.model.to(device)

    def __call__(self, input_data, batch_size=8):
        all_predicted = []
        num_batches = (len(input_data) + batch_size - 1) // batch_size
        for step in tqdm(range(num_batches), desc='Extracting phrases'):
            batch_start = step * batch_size
            batch_end = min((step + 1) * batch_size, len(input_data))
            batch_text = [entry['context'] for entry in input_data[batch_start:batch_end]]
            inputs = self.tokenizer.batch_encode_plus(
                batch_text,
                padding='max_length',
                truncation='longest_first',
                max_length=self.max_seq_length,
                return_offsets_mapping=True,
                return_tensors='pt',
            )
            offset_mappings = inputs['offset_mapping'].detach().cpu().numpy()

            del inputs['offset_mapping']
            for key, value in inputs.items():
                inputs[key] = value.to(self.model.device)
            outputs = self.model(**inputs)
            phrase_logits, start_logits, end_logits = outputs[0], outputs[1], outputs[2]
            start_predicted = np.argmax(start_logits.detach().cpu().numpy(), axis=-1)
            end_predicted = np.argmax(end_logits.detach().cpu().numpy(), axis=-1)
            phrase_predicted = np.argmax(phrase_logits.detach().cpu().numpy(), axis=-1)
            all_predicted.extend(generate_outputs(offset_mappings, start_predicted, end_predicted, phrase_predicted))

        for i in range(len(input_data)):
            input_data[i]['predicted'] = [
                (input_data[i]['context'][start:end], start, end) for start, end in all_predicted[i]
            ]

        return input_data
