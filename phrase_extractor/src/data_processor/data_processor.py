# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/7/26
@Desc               : 
@Last modified by   : Bao
@Last modified date : 2020/9/9
"""

import os
import copy
import json
import torch
import logging
from tqdm import tqdm
from torch.utils.data import TensorDataset

from src.utils import read_json_lines

logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, guid, context, phrase_spans=None):
        self.guid = guid
        self.context = context
        self.phrase_spans = phrase_spans

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    def __init__(self, guid, input_ids, attention_mask=None, token_type_ids=None,
                 start_labels=None, end_labels=None, phrase_labels=None):
        self.guid = guid
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.start_labels = start_labels
        self.end_labels = end_labels
        self.phrase_labels = phrase_labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def convert_examples_to_features(examples, tokenizer, max_length=512):
    features = []
    for (ex_index, example) in enumerate(tqdm(examples, desc="Converting Examples")):
        encoded = tokenizer.encode_plus(
            example.context,
            padding="max_length",
            truncation="longest_first",
            max_length=max_length,
            return_offsets_mapping=True,
        )

        encoded["guid"] = example.guid
        start_labels = [0] * max_length
        end_labels = [0] * max_length
        phrase_labels = [[0] * max_length for _ in range(max_length)]
        for char_start, char_end in example.phrase_spans:
            token_start, token_end = -1, -1
            for i, (start, end) in enumerate(encoded["offset_mapping"]):
                if start <= char_start < end:
                    token_start = i
                if start < char_end <= end:
                    token_end = i
            if token_start == -1 or token_end == -1 or token_start > token_end:
                continue
            start_labels[token_start] = 1
            end_labels[token_end] = 1
            phrase_labels[token_start][token_end] = 1
        encoded["start_labels"] = start_labels
        encoded["end_labels"] = end_labels
        encoded["phrase_labels"] = phrase_labels

        del encoded["offset_mapping"]
        features.append(InputFeatures(**encoded))

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: {}".format(encoded["guid"]))
            logger.info("input_ids: {}".format(encoded["input_ids"]))
            logger.info("attention_mask: {}".format(encoded["attention_mask"]))
            logger.info("token_type_ids: {}".format(encoded["token_type_ids"]))
            logger.info("start_labels: {}".format(encoded["start_labels"]))
            logger.info("end_labels: {}".format(encoded["end_labels"]))

            # logger.info('=' * 20)
            # for start, end in example.phrase_spans:
            #     logger.info(example.context[start:end])
            #
            # logger.info('=' * 20)
            # for start in range(max_length):
            #     for end in range(start, max_length):
            #         if phrase_labels[start][end] == 1:
            #             logger.info(tokenizer.decode(encoded['input_ids'][start:end + 1]))

    return features


class DataProcessor:
    def __init__(
        self,
        model_type,
        model_name_or_path,
        max_seq_length,
        data_dir="",
        overwrite_cache=False
    ):
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length

        self.data_dir = data_dir
        self.cache_dir = os.path.join(data_dir, "cache_extractor")
        self.overwrite_cache = overwrite_cache

    def load_and_cache_data(self, role, tokenizer):
        cached_file = os.path.join(
            self.cache_dir,
            "cached_{}_{}_{}".format(
                role,
                list(filter(None, self.model_name_or_path.split("/"))).pop(),
                str(self.max_seq_length),
            ),
        )

        if os.path.exists(cached_file) and not self.overwrite_cache:
            logger.info("Loading features from cached file {}".format(cached_file))
            features = torch.load(cached_file)
        else:
            examples = []
            for line in tqdm(
                list(read_json_lines(os.path.join(self.data_dir, "data_{}.json".format(role)))),
                desc="Loading Examples"
            ):
                sample = {'guid': len(examples)}
                sample.update(self._load_line(line))
                examples.append(InputExample(**sample))
            features = convert_examples_to_features(examples, tokenizer, self.max_seq_length)

            logger.info("Saving features into cached file {}".format(cached_file))
            os.makedirs(self.cache_dir, exist_ok=True)
            torch.save(features, cached_file)

        return self._create_tensor_dataset(features)

    def _create_tensor_dataset(self, features, do_predict=False):
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use token_type_ids
        if self.model_type in ["bert", "bert-pu", "xlnet", "albert"]:
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        else:
            all_token_type_ids = torch.tensor([[0] * self.max_seq_length for _ in features], dtype=torch.long)

        if not do_predict:
            all_start_labels = torch.tensor([f.start_labels for f in features], dtype=torch.long)
            all_end_labels = torch.tensor([f.end_labels for f in features], dtype=torch.long)
            all_phrase_labels = torch.tensor([f.phrase_labels for f in features], dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids, all_attention_mask, all_token_type_ids,
                all_start_labels, all_end_labels, all_phrase_labels,
            )
        else:
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)

        return dataset

    def _load_line(self, line):
        context = line["context"]
        qas = line["qas"]

        phrase_spans = []
        for qa in qas:
            phrase_spans.append((qa["answer_start"], qa["answer_end"]))

        return {
            "context": context,
            "phrase_spans": phrase_spans,
        }
