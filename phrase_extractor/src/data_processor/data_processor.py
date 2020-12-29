# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/7/26
@Desc               : 
@Last modified by   : Bao
@Last modified date : 2020/12/29
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

from src.utils import read_json_lines, pad_batch

logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, guid, context, phrase_spans):
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
    def __init__(self, guid, input_ids, attention_mask=None, token_type_ids=None, src_index=None, tgt_index=None,
                 offset_mapping=None, start_labels=None, end_labels=None, phrase_labels=None):
        self.guid = guid
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.src_index = src_index
        self.tgt_index = tgt_index
        self.offset_mapping = offset_mapping
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


def initialize_pyramid_tree(attention_mask, seq_length):
    src_index = []
    tgt_index = []
    for i in range(seq_length):
        for j in range(i, seq_length):
            src = i * seq_length + j

            # edges to siblings
            if i == j:
                ii, jj = i - 1, j - 1
                if 0 <= ii < seq_length and 0 <= jj < seq_length and attention_mask[ii] * attention_mask[jj] != 0:
                    tgt = ii * seq_length + jj
                    src_index.append(src)
                    tgt_index.append(tgt)
                ii, jj = i + 1, j + 1
                if 0 <= ii < seq_length and 0 <= jj < seq_length and attention_mask[ii] * attention_mask[jj] != 0:
                    tgt = ii * seq_length + jj
                    src_index.append(src)
                    tgt_index.append(tgt)

            # edges to parent
            ii, jj = i, j + 1
            if 0 <= ii < seq_length and 0 <= jj < seq_length and attention_mask[ii] * attention_mask[jj] != 0:
                tgt = ii * seq_length + jj
                src_index.append(src)
                tgt_index.append(tgt)
            ii, jj = i - 1, j
            if 0 <= ii < seq_length and 0 <= jj < seq_length and attention_mask[ii] * attention_mask[jj] != 0:
                tgt = ii * seq_length + jj
                src_index.append(src)
                tgt_index.append(tgt)

    return src_index, tgt_index


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

        # initialize tree
        encoded["src_index"], encoded["tgt_index"] = initialize_pyramid_tree(encoded["attention_mask"], max_length)

        # initialize labels
        start_labels = [0] * max_length
        end_labels = [0] * max_length
        phrase_labels = [[0] * max_length for _ in range(max_length)]
        for _, phrase_start, phrase_end in example.phrase_spans:
            start_index, end_index = -1, -1
            for i, (start, end) in enumerate(encoded["offset_mapping"]):
                if start <= phrase_start < end: start_index = i
                if start < phrase_end <= end: end_index = i
            if start_index == -1 or end_index == -1 or start_index > end_index: continue
            start_labels[start_index] = 1
            end_labels[end_index] = 1
            phrase_labels[start_index][end_index] = 1
        encoded["start_labels"] = start_labels
        encoded["end_labels"] = end_labels
        encoded["phrase_labels"] = coo_matrix(phrase_labels).reshape(1, max_length * max_length)

        features.append(InputFeatures(**encoded))

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: {}".format(encoded["guid"]))
            logger.info("input_ids: {}".format(encoded["input_ids"]))
            logger.info("start_labels: {}".format(encoded["start_labels"]))
            logger.info("end_labels: {}".format(encoded["end_labels"]))
            logger.info("tokens: {}".format(tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)))
            logger.info("phrases: {}".format([v[0] for v in example.phrase_spans]))

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

        all_src_index = torch.tensor(pad_batch([f.src_index for f in features], 0), dtype=torch.long)
        all_tgt_index = torch.tensor(pad_batch([f.tgt_index for f in features], 0), dtype=torch.long)

        all_offset_mappings = torch.tensor([f.offset_mapping for f in features], dtype=torch.long)

        all_start_labels = torch.tensor([f.start_labels for f in features], dtype=torch.long)
        all_end_labels = torch.tensor([f.end_labels for f in features], dtype=torch.long)
        all_phrase_labels = vstack([f.phrase_labels for f in features])
        all_phrase_labels = torch.sparse_coo_tensor(
            torch.tensor(np.vstack([all_phrase_labels.row, all_phrase_labels.col]), dtype=torch.long),
            torch.tensor(all_phrase_labels.data, dtype=torch.long),
            size=all_phrase_labels.shape,
            dtype=torch.long,
        )
        dataset = TensorDataset(
            all_input_ids, all_attention_mask, all_token_type_ids, all_src_index, all_tgt_index,
            all_offset_mappings, all_start_labels, all_end_labels, all_phrase_labels,
        )

        return dataset

    def _load_line(self, line):
        context = line["context"]
        qas = line["qas"]

        phrase_spans = [(qa['answer'], qa["answer_start"], qa["answer_end"]) for qa in qas]

        return {"context": context, "phrase_spans": phrase_spans}
