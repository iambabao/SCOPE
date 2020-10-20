# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2020/10/12
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/10/20
"""

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


class Reader:
    """

    Example:
        input_data = [
            {'context': 'My name is Sarah.', 'question': 'Where do I live?'},
            {'context': 'My name is Sarah and I live in London.', 'question': 'Where do I live?'},
            {'context': 'Sarah lived in London. Jone lived in Canada.', 'question': 'Where do Sarah live?'},
            {'context': 'Sarah lived in London. Jone lived in Canada.', 'question': 'Where do William live?'},
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
            cache_dir=cache_dir if cache_dir else None,
        )
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir if cache_dir else None,
        )
        self.model.to(device)
        self.model.eval()

    def __call__(self, input_data, max_length=None):
        batch_text_pairs = [(entry['question'], entry['context'].replace('<hl>', '')) for entry in input_data]
        inputs = self.tokenizer.batch_encode_plus(
            batch_text_pairs,
            padding='max_length',
            truncation='longest_first',
            max_length=max_length,
            return_length=True,
            return_tensors='pt',
        )
        input_ids = inputs['input_ids'].detach().cpu().tolist()
        input_lengths = inputs['length'].detach().cpu().tolist()
        del inputs['length']
        for key, value in inputs.items():
            inputs[key] = value.to(self.model.device)

        outputs = self.model(**inputs)
        answer_start_scores, answer_end_scores = outputs[0], outputs[1]
        answer_starts = torch.argmax(answer_start_scores, dim=-1).detach().cpu().tolist()
        answer_ends = (torch.argmax(answer_end_scores, dim=-1) + 1).detach().cpu().tolist()

        for i, (ids, length, start, end) in enumerate(zip(input_ids, input_lengths, answer_starts, answer_ends)):
            sep_index = ids.index(self.tokenizer.sep_token_id)
            if start >= end or start <= sep_index or end <= sep_index or start > length or end > length:
                answer = ''
            else:
                answer = self.tokenizer.decode(ids[start:end], skip_special_tokens=True)
            input_data[i]['predicted_answer'] = answer
            # input_data[i]['input_ids'] = ids[:length]
            # input_data[i]['predicted_start'] = start
            # input_data[i]['predicted_end'] = end

        return input_data
