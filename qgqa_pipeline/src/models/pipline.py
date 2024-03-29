# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2020/10/14
@Desc               :
@Last modified by   : Bao
@Last modified date : 2021/5/11
"""

from .generator import Generator
from .reader import Reader


class Pipeline:
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
        pipeline = Pipeline(
            generator_model='valhalla/t5-base-qg-hl',
            reader_model='deepset/roberta-large-squad2',
            cache_dir='your_cache_dir',
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )

        results = pipeline(input_data, beam_size=5)
        print(json.dumps(results, ensure_ascii=False, indent=4))
    """

    def __init__(self, generator_model, reader_model, do_lower_case=False, cache_dir=None, device='cpu'):
        self.generator = Generator(generator_model, do_lower_case=do_lower_case, cache_dir=cache_dir, device=device)
        self.reader = Reader(reader_model, do_lower_case=do_lower_case, cache_dir=cache_dir, device=device)

    def __call__(self, input_data, max_seq_length=None, batch_size=8, beam_size=1, temperature=1.0):
        results = self.generator(input_data, max_seq_length=max_seq_length, batch_size=batch_size, beam_size=beam_size)
        results = self.reader(results, max_seq_length=max_seq_length, batch_size=batch_size, temperature=temperature)

        return results
