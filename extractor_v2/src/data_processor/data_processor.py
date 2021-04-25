# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/7/26
@Desc               : 
@Last modified by   : Bao
@Last modified date : 2021/4/25
"""

import os
import copy
import json
import torch
import logging
import numpy as np
from tqdm import tqdm
from scipy.sparse import coo_matrix, vstack
from torch.utils.data import TensorDataset

from src.utils import read_json_lines, pad_list

logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, guid, context, token_spans, pos_tag, phrase_spans):
        self.guid = guid
        self.context = context
        self.token_spans = token_spans
        self.pos_tag = pos_tag
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
                 token_mapping=None, pos_tag=None, num_tokens=None, phrase_labels=None):
        self.guid = guid
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.token_mapping = token_mapping
        self.pos_tag = pos_tag
        self.num_tokens = num_tokens
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


def convert_examples_to_features(examples, tokenizer, max_seq_length, max_num_tokens):
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

        # initialize features
        token_mapping = [max_num_tokens - 1] * max_seq_length
        for i, (start, end) in enumerate(encoded["offset_mapping"]):
            for j, (token, token_start, token_end) in enumerate(example.token_spans[:max_num_tokens]):
                if token_start <= start < end <= token_end:
                    token_mapping[i] = j
        encoded["token_mapping"] = token_mapping
        encoded["pos_tag"] = pad_list(example.pos_tag, 16, max_seq_length)  # 16 is X
        encoded["num_tokens"] = min(len(example.token_spans), max_num_tokens)

        # initialize labels
        phrase_labels = [[0] * max_num_tokens for _ in range(max_num_tokens)]
        for _, phrase_start, phrase_end in example.phrase_spans:
            if phrase_start >= max_num_tokens or phrase_end >= max_num_tokens: continue
            phrase_labels[phrase_start][phrase_end] = 1
        encoded["phrase_labels"] = coo_matrix(phrase_labels).reshape(1, max_num_tokens * max_num_tokens)

        del encoded["offset_mapping"]
        features.append(InputFeatures(**encoded))

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: {}".format(encoded["guid"]))
            logger.info("input_ids: {}".format(encoded["input_ids"]))
            logger.info("token_mapping: {}".format(encoded["token_mapping"]))
            logger.info("tokens: {}".format([v[0] for v in example.token_spans]))
            logger.info("phrases: {}".format([v[0] for v in example.phrase_spans]))

    return features


class DataProcessor:
    def __init__(
            self,
            model_type,
            model_name_or_path,
            max_seq_length,
            max_num_tokens,
            data_dir="",
            overwrite_cache=False
    ):
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.max_num_tokens = max_num_tokens

        self.data_dir = data_dir
        self.cache_dir = os.path.join(data_dir, "cache_v2")
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
                list(read_json_lines(os.path.join(self.data_dir, "data_{}.feature.json".format(role)))),
                desc="Loading Examples"
            ):
                sample = {'guid': len(examples)}
                sample.update(self._load_line(line))
                examples.append(InputExample(**sample))
            logger.info("Saving examples into cached file {}".format(cached_examples))
            torch.save(examples, cached_examples)

        cached_features = os.path.join(
            self.cache_dir,
            "cached_feature_{}_{}_{}".format(
                role,
                list(filter(None, self.model_name_or_path.split("/"))).pop(),
                str(self.max_seq_length),
            ),
        )
        if os.path.exists(cached_features) and not self.overwrite_cache:
            logger.info("Loading features from cached file {}".format(cached_features))
            features = torch.load(cached_features)
        else:
            features = convert_examples_to_features(examples, tokenizer, self.max_seq_length, self.max_num_tokens)
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

        all_token_mapping = torch.tensor([f.token_mapping for f in features], dtype=torch.long)
        all_pos_tag = torch.tensor([f.pos_tag for f in features], dtype=torch.long)
        all_num_tokens = torch.tensor([f.num_tokens for f in features], dtype=torch.long)

        all_phrase_labels = vstack([f.phrase_labels for f in features])
        all_phrase_labels = torch.sparse_coo_tensor(
            torch.tensor(np.vstack([all_phrase_labels.row, all_phrase_labels.col]), dtype=torch.long),
            torch.tensor(all_phrase_labels.data, dtype=torch.long),
            size=all_phrase_labels.shape,
            dtype=torch.long,
        )

        dataset = TensorDataset(
            all_input_ids, all_attention_mask, all_token_type_ids,
            all_token_mapping, all_pos_tag, all_num_tokens, all_phrase_labels,
        )

        return dataset

    def _load_line(self, line):
        token_spans = []
        for v in line['token_spans']: token_spans.extend(v)
        pos_tag = []
        for v in line['pos_tag']: pos_tag.extend(v)

        return {
            'context': line['context'],
            'token_spans': token_spans,
            'pos_tag': pos_tag,
            'phrase_spans': line['phrase_spans']
        }
