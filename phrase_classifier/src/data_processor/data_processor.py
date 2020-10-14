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

from src.utils import read_json_lines, pad_batch

logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, guid, context, phrase_spans, labels=None):
        self.guid = guid
        self.context = context
        self.phrase_spans = phrase_spans
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


class InputFeatures(object):
    def __init__(self, guid, phrase_spans, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        self.guid = guid
        self.phrase_spans = phrase_spans
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
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


def convert_examples_to_features(examples, tokenizer, max_length=512):
    features = []
    for (ex_index, example) in enumerate(tqdm(examples, desc="Converting Examples")):
        input_ids = []
        index_mapping = {0: 0}
        for i, token in enumerate(example.context):
            ids = tokenizer.encode(token if i == 0 else ' ' + token, add_special_tokens=False)  # RoBERTa needs blank
            input_ids.extend(ids)
            index_mapping[i + 1] = index_mapping[i] + len(ids)

        encoded = {"guid": example.guid}
        encoded.update(tokenizer.prepare_for_model(
            input_ids,
            padding="max_length",
            truncation="longest_first",
            max_length=max_length,
        ))
        phrase_spans = [(index_mapping[start], index_mapping[end]) for start, end in example.phrase_spans]
        for i, _id in enumerate(encoded['input_ids']):
            if _id in tokenizer.all_special_ids:
                for j, (start, end) in enumerate(phrase_spans):
                    if start >= i: start += 1
                    if end >= i: end += 1
                    phrase_spans[j] = (start, end)
        truncated_phrase_spans = []
        truncated_labels = []
        for (start, end), label in zip(phrase_spans, example.labels):
            if start < max_length and end < max_length:
                truncated_phrase_spans.append((start, end))
                truncated_labels.append(label)
        assert len(truncated_phrase_spans) == len(truncated_labels)
        encoded["phrase_spans"] = truncated_phrase_spans
        encoded["labels"] = truncated_labels
        features.append(InputFeatures(**encoded))

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: {}".format(example.guid))
            logger.info("features: {}".format(encoded))

            # logger.info('=' * 20)
            # for start, end in example.phrase_spans:
            #     logger.info(' '.join(example.context[start:end]))
            #
            # logger.info('=' * 20)
            # for start, end in example.phrase_spans:
            #     start, end = index_mapping[start], index_mapping[end]
            #     logger.info(tokenizer.decode(input_ids[start:end]))
            #
            # logger.info('=' * 20)
            # for start, end in phrase_spans:
            #     logger.info(tokenizer.decode(encoded['input_ids'][start:end]))

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
        self.cache_dir = os.path.join(data_dir, "cache_classifier")
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
                list(read_json_lines(os.path.join(self.data_dir, "{}.tokenized.json".format(role)))),
                desc="Loading Examples"
            ):
                sample = {'guid': len(examples)}
                sample.update(self._load_line(line))
                examples.append(InputExample(**sample))
            features = convert_examples_to_features(examples, tokenizer, self.max_seq_length)

            logger.info("Saving features into cached file {}".format(cached_file))
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

        all_phrase_spans = torch.tensor(pad_batch([f.phrase_spans for f in features], (0, 1)), dtype=torch.long)
        all_num_phrases = torch.tensor([len(f.phrase_spans) for f in features], dtype=torch.long)

        if not do_predict:
            all_labels = torch.tensor(pad_batch([f.labels for f in features], -1), dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids, all_attention_mask, all_token_type_ids,
                all_phrase_spans, all_num_phrases, all_labels,
            )
        else:
            dataset = TensorDataset(
                all_input_ids, all_attention_mask, all_token_type_ids,
                all_phrase_spans, all_num_phrases,
            )

        return dataset

    def _load_line(self, line):
        context = line["context"]
        phrase_spans = []
        labels = []
        for start, end, label in line["phrases"]:
            phrase_spans.append((start, end))
            labels.append(min(1, label))

        return {
            "context": context,
            "phrase_spans": phrase_spans,
            "labels": labels,
        }
