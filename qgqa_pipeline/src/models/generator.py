# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2020/10/13
@Desc               :
@Last modified by   : Bao
@Last modified date : 2021/5/11
"""

import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Generator:
    """
    Example:
        import json
        import torch

        input_data = [
            {'context': 'My name is Sarah.', 'answer': ('Sarah', 11, 16)},
            {'context': 'My name is Sarah and I live in London.', 'answer': ('London', 31, 37)},
            {'context': 'Sarah lived in London. Jone lived in Canada.', 'answer': ('Canada', 37, 43)},
            {'context': 'Sarah lived in London. Jone lived in Canada.', 'answer': ('lived', 28, 33)},
        ]
        generator = Generator(
            'valhalla/t5-base-qg-hl',
            'your_cache_dir',
            'cuda' if torch.cuda.is_available() else 'cpu',
        )

        results = generator(input_data, beam_size=5)
        print(json.dumps(results, ensure_ascii=False, indent=4))
    """

    def __init__(self, model_name_or_path, do_lower_case=False, cache_dir=None, device='cpu'):
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            do_lower_case=do_lower_case,
            cache_dir=cache_dir if cache_dir else None,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir if cache_dir else None,
        )
        self.model.to(device)
        self.model.eval()

    def convert_dataset(self, input_data, max_seq_length):
        """
        Convert inputs into dataset based on https://github.com/patil-suraj/question_generation

        Args:
            input_data:
            max_seq_length:

        Returns:

        """

        encoded_data = []
        for entry in tqdm(input_data, desc='Converting dataset'):
            context = entry['context']
            answer, answer_start, answer_end = entry['answer']
            text = context[:answer_start] + '<hl> ' + answer + ' <hl>' + context[answer_end:]
            text = 'generate question: {}'.format(text)
            encoded = self.tokenizer.encode_plus(
                text,
                padding='max_length',
                truncation='longest_first',
                max_length=max_seq_length,
            )
            encoded_data.append(encoded)

        all_input_ids = torch.tensor([v["input_ids"] for v in encoded_data], dtype=torch.long)
        all_attention_mask = torch.tensor([v["attention_mask"] for v in encoded_data], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_attention_mask)
        return dataset

    def __call__(self, input_data, max_seq_length, batch_size=8, beam_size=1, *args, **kwargs):
        dataset = self.convert_dataset(input_data, max_seq_length)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

        all_ids_with_beam = []
        for batch in tqdm(dataloader, desc='Generating questions'):
            with torch.no_grad():
                inputs = {
                    'input_ids': batch[0].to(self.model.device),
                    'attention_mask': batch[1].to(self.model.device),
                }

                ids_with_beam = self.model.generate(**inputs, num_beams=beam_size, num_return_sequences=beam_size)
                ids_with_beam = ids_with_beam.reshape([-1, beam_size, ids_with_beam.shape[-1]])
                all_ids_with_beam.extend(ids_with_beam.detach().cpu().tolist())

        for i, entry in enumerate(tqdm(input_data, desc='Decoding outputs')):
            questions = self.tokenizer.batch_decode(all_ids_with_beam[i], skip_special_tokens=True)
            entry['questions'] = questions

        return input_data
