# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2020/10/12
@Desc               :
@Last modified by   : Bao
@Last modified date : 2021/5/11
"""

import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


class Reader:
    """
    Example:
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
            'deepset/roberta-large-squad2',
            'your_cache_dir',
            'cuda' if torch.cuda.is_available() else 'cpu',
        )

        results = reader(input_data)
        print(json.dumps(results, ensure_ascii=False, indent=4))
    """

    def __init__(self, model_name_or_path, do_lower_case=False, cache_dir=None, device='cpu'):
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            do_lower_case=do_lower_case,
            use_fast=True,
            cache_dir=cache_dir if cache_dir else None,
        )
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir if cache_dir else None,
        )
        self.model.to(device)
        self.model.eval()

    def convert_dataset(self, input_data, max_seq_length):
        encoded_data, answer_indexes = [], []
        for entry in tqdm(input_data, desc='Converting dataset'):
            encoded = self.tokenizer.encode_plus(
                entry['question'], entry['context'],
                padding='max_length',
                truncation='longest_first',
                max_length=max_seq_length,
                return_offsets_mapping=True,
            )
            encoded_data.append(encoded)

            token_start, token_end = 0, 0
            # We keep the last match because the offset mapping of context is behind the question
            for i in range(len(encoded['offset_mapping']) - 1):
                if encoded['offset_mapping'][i][0] <= entry['answer'][1] < encoded['offset_mapping'][i + 1][0]:
                    token_start = i
                if encoded['offset_mapping'][i][1] <= entry['answer'][2] <= encoded['offset_mapping'][i + 1][0]:
                    token_end = i
            answer_indexes.append([token_start, token_end])

        all_input_ids = torch.tensor([v["input_ids"] for v in encoded_data], dtype=torch.long)
        all_attention_mask = torch.tensor([v["attention_mask"] for v in encoded_data], dtype=torch.long)
        # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use token_type_ids
        if "roberta" not in self.model_name_or_path:
            all_token_type_ids = torch.tensor([v["token_type_ids"] for v in encoded_data], dtype=torch.long)
        else:
            all_token_type_ids = torch.tensor([[0] * max_seq_length for _ in encoded_data], dtype=torch.long)
        all_answer_indexes = torch.tensor(answer_indexes, dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_answer_indexes)
        return dataset, answer_indexes

    def __call__(self, input_data, max_seq_length, batch_size=8, temperature=1.0, *args, **kwargs):
        expanded_data = []
        for entry in input_data:
            for question in entry['questions']:
                expanded_data.append({'context': entry['context'], 'question': question, 'answer': entry['answer']})

        dataset, answer_indexes = self.convert_dataset(expanded_data, max_seq_length)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

        all_scores = []
        for batch in tqdm(dataloader, desc='Scoring answer'):
            with torch.no_grad():
                inputs = {
                    'input_ids': batch[0].to(self.model.device),
                    'attention_mask': batch[1].to(self.model.device),
                }
                # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use token_type_ids
                if "roberta" not in self.model_name_or_path:
                    inputs["token_type_ids"] = batch[2].to(self.model.device)

                outputs = self.model(**inputs)
                start_scores, end_scores = outputs[0], outputs[1]
                start_scores = torch.softmax(start_scores / temperature, dim=-1).detach().cpu().tolist()
                end_scores = torch.softmax(end_scores / temperature, dim=-1).detach().cpu().tolist()

                for s_score, e_score, (start, end) in zip(start_scores, end_scores, batch[-1]):
                    all_scores.append((s_score[start], e_score[end]))

        num_questions = len(input_data[0]['questions'])
        for i in range(len(input_data)):
            start = i * num_questions
            end = (i + 1) * num_questions
            input_data[i]['qa_scores'] = all_scores[start:end]

        return input_data
