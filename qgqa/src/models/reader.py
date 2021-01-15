# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2020/10/12
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/12/8
"""

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


class Reader:
    """
    Examples:
        import json

        input_data = [
            {'context': 'My name is Sarah.',
             'answer': ('Sarah', 11, 16),
             'questions': ['Where do I live?']},
            {'context': 'My name is Sarah and I live in London.',
             'answer': ('London', 31, 37),
             'questions': ['Where do I live?']},
            {'context': 'Sarah lived in London. Jone lived in Canada.',
             'answer': ('Canada', 37, 43),
             'questions': ['Where do Sarah live?']},
            {'context': 'Sarah lived in London. Jone lived in Canada.',
             'answer': ('Canada', 37, 43),
             'questions': ['Where do William live?']},
        ]
        reader = Reader(
            'deepset/bert-base-cased-squad2',
            'your_cache_dir',
            'cuda' if torch.cuda.is_available() else 'cpu',
        )

        results = reader(input_data)
        print(json.dumps(results, ensure_ascii=False, indent=4))
    """

    def __init__(self, model_name_or_path, cache_dir=None, device='cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=True,
            cache_dir=cache_dir if cache_dir else None,
        )
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir if cache_dir else None,
        )
        self.model.to(device)
        self.model.eval()

    def __call__(self, input_data, max_length=None, batch_size=8):
        beam_size = len(input_data[0]['questions'])

        _input_data = []
        for entry in input_data:
            for question in entry['questions']:
                _input_data.append({'context': entry['context'], 'question': question, 'answer': entry['answer']})
        all_scores = []
        num_batches = (len(_input_data) + batch_size - 1) // batch_size
        for step in tqdm(range(num_batches), desc='Calculating scores'):
            batch_start = step * batch_size
            batch_end = min((step + 1) * batch_size, len(_input_data))
            batch_text_pairs = []
            for entry in _input_data[batch_start:batch_end]:
                batch_text_pairs.append((entry['question'], entry['context']))
            inputs = self.tokenizer.batch_encode_plus(
                batch_text_pairs,
                padding='max_length',
                truncation='longest_first',
                max_length=max_length,
                return_offsets_mapping=True,
                return_tensors='pt',
            )
            offset_mappings = inputs['offset_mapping'].detach().cpu().tolist()

            answer_indices = []
            for mappings, entry in zip(offset_mappings, _input_data[batch_start:batch_end]):
                token_start, token_end = -1, -1
                # We keep the last match because the offset mapping of context is behind the question
                for i in range(len(mappings) - 1):
                    if mappings[i][0] <= entry['answer'][1] < mappings[i + 1][0]: token_start = i
                    if mappings[i][1] <= entry['answer'][2] <= mappings[i + 1][0]: token_end = i
                answer_indices.append((token_start, token_end))
            assert len(answer_indices) == len(batch_text_pairs)

            del inputs['offset_mapping']
            for key, value in inputs.items():
                inputs[key] = value.to(self.model.device)
            outputs = self.model(**inputs)
            start_scores, end_scores = outputs[0], outputs[1]
            start_scores = torch.softmax(start_scores, dim=-1).detach().cpu().tolist()
            end_scores = torch.softmax(end_scores, dim=-1).detach().cpu().tolist()

            for i, (start_score, end_score, (start, end)) in enumerate(zip(start_scores, end_scores, answer_indices)):
                if start == --1 or end == -1:
                    all_scores.append((max(start_score), max(end_score)))
                else:
                    all_scores.append((start_score[start], end_score[end]))

        for i in range(len(input_data)):
            start = i * beam_size
            end = (i + 1) * beam_size
            input_data[i]['scores'] = all_scores[start:end]

        return input_data
