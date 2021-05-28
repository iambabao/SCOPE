# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/7/26
@Desc               : 
@Last modified by   : Bao
@Last modified date : 2021/1/18
"""

import os
import copy
import json
import torch
import logging
from tqdm import tqdm
from torch.utils.data import TensorDataset

from src.utils import read_json_lines, pad_list

logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, guid, context, phrases, labels, golden):
        self.guid = guid
        self.context = context
        self.phrases = phrases
        self.labels = labels
        self.golden = golden

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
                 mappings=None, num_phrases=None, labels=None):
        self.guid = guid
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.mappings = mappings
        self.num_phrases = num_phrases
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def convert_examples_to_features(examples, tokenizer, max_seq_length):
    features = []
    for (ex_index, example) in enumerate(tqdm(examples, desc="Converting Examples")):
        encoded = {"guid": example.guid}

        encoded.update(tokenizer.encode_plus(
            example.context,
            padding="max_length",
            truncation="longest_first",
            max_length=max_seq_length,
            return_offsets_mapping=True,
        ))

        mappings = []
        for _, phrase_start, phrase_end in example.phrases:
            token_start, token_end = -1, -1
            for i in range(len(encoded["offset_mapping"]) - 1):
                if encoded["offset_mapping"][i][0] <= phrase_start < encoded["offset_mapping"][i + 1][0]:
                    token_start = i
                if encoded["offset_mapping"][i][1] <= phrase_end <= encoded["offset_mapping"][i + 1][1]:
                    token_end = i
            if token_start != -1 and token_end != -1: mappings.append((token_start, token_end))
        encoded["mappings"] = mappings
        encoded["num_phrases"] = len(mappings)
        encoded["labels"] = example.labels

        del encoded["offset_mapping"]
        features.append(InputFeatures(**encoded))

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: {}".format(encoded["guid"]))
            logger.info("input_ids: {}".format(encoded["input_ids"]))
            logger.info("phrases: {}".format([
                tokenizer.decode(encoded["input_ids"][start:end + 1], skip_special_tokens=True)
                for start, end in encoded["mappings"]
            ]))
            logger.info("labels: {}".format(encoded["labels"]))

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
        self.cache_dir = os.path.join(data_dir, "cache")
        self.overwrite_cache = overwrite_cache

    def load_and_cache_data(self, role, tokenizer):
        os.makedirs(self.cache_dir, exist_ok=True)

        cached_examples = os.path.join(self.cache_dir, "cached_example_{}".format(role))
        if os.path.exists(cached_examples) and not self.overwrite_cache:
            logger.info("Loading examples from cached file {}".format(cached_examples))
            examples = torch.load(cached_examples)
        else:
            examples = []
            for line in tqdm(
                list(read_json_lines(os.path.join(self.data_dir, "data_{}.json".format(role)))),
                desc="Loading Examples"
            ):
                sample = {'guid': len(examples)}
                sample.update(line)
                examples.append(InputExample(**sample))
            logger.info("Saving examples into cached file {}".format(cached_examples))
            torch.save(examples, cached_examples)

        cached_features = os.path.join(
            self.cache_dir,
            "cached_feature_{}_{}_{}".format(
                role,
                list(filter(None, self.model_name_or_path.split("/"))).pop(),
                self.max_seq_length,
            ),
        )
        if os.path.exists(cached_features) and not self.overwrite_cache:
            logger.info("Loading features from cached file {}".format(cached_features))
            features = torch.load(cached_features)
        else:
            features = convert_examples_to_features(examples, tokenizer, self.max_seq_length)
            logger.info("Saving features into cached file {}".format(cached_features))
            torch.save(features, cached_features)

        return examples, self._create_tensor_dataset(features)

    def _create_tensor_dataset(self, features):
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use token_type_ids
        if self.model_type in ["bert", "xlnet", "albert"]:
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        else:
            all_token_type_ids = torch.tensor([[0] * self.max_seq_length for _ in features], dtype=torch.long)

        all_num_phrases = torch.tensor([f.num_phrases for f in features], dtype=torch.long)
        max_num_phrases = int(torch.max(all_num_phrases))
        all_mappings = torch.tensor([pad_list(f.mappings, [0, 0], max_num_phrases) for f in features], dtype=torch.long)
        all_labels = torch.tensor([pad_list(f.labels, 0, max_num_phrases) for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_mappings, all_num_phrases, all_labels)
        return dataset
