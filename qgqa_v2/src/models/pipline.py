# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2020/10/14
@Desc               :
@Last modified by   : Bao
@Last modified date : 2021/1/15
"""

from .generator import Generator
from .reader import Reader


class Pipeline:
    """
    Examples:
        import json
        import torch

        input_data = [
            {'context': 'My name is Sarah.', 'answer': ('Sarah', 11, 16)},
            {'context': 'My name is Sarah and I live in London.', 'answer': ('London', 31, 37)},
            {'context': 'Sarah lived in London. Jone lived in Canada.', 'answer': ('Canada', 37, 43)},
            {'context': 'Sarah lived in London. Jone lived in Canada.', 'answer': ('lived', 28, 33)},
        ]
        pipeline = Pipeline(
            generator_model_name='valhalla/t5-base-qg-hl',
            reader_model_name='deepset/bert-base-cased-squad2',
            cache_dir='your_cache_dir',
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )

        results = pipeline(input_data, beam_size=5)
        print(json.dumps(results, ensure_ascii=False, indent=4))
    """

    def __init__(self, generator_model_name, reader_model_name, cache_dir=None, device='cpu'):
        self.generator = Generator(generator_model_name, cache_dir=cache_dir, device=device)
        self.reader = Reader(reader_model_name, cache_dir=cache_dir, device=device)

    def __call__(self, input_data, max_length=None, batch_size=8, beam_size=1):
        results = self.generator(input_data, max_length=max_length, batch_size=batch_size, beam_size=beam_size)
        results = self.reader(results, max_length=max_length, batch_size=batch_size, beam_size=beam_size)

        return results
