# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2020/10/14
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/10/14
"""

from .generator import Generator
from .reader import Reader


class Pipeline:
    """

    Example:
        input_data = [
            {'context': 'My name is <hl> Sarah <hl> .', 'raw_answer': 'Sarah'},
            {'context': 'My name is Sarah and I live in <hl> London <hl> .', 'raw_answer': 'London'},
            {'context': 'Sarah lived in London. Jone lived in <hl> Canada <hl> .', 'raw_answer': 'Canada'},
            {'context': 'Sarah <hl> lived <hl> in London. Jone lived in Canada.', 'raw_answer': 'lived'},
        ]
        pipeline = Pipeline(
            generator_model_name='valhalla/t5-base-qg-hl',
            reader_model_name='deepset/bert-base-cased-squad2',
            cache_dir='your_cache_dir',
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )

        results = pipeline(input_data)
        print(json.dumps(results, ensure_ascii=False, indent=4))
    """

    def __init__(self, generator_model_name, reader_model_name, cache_dir=None, device='cpu'):
        self.generator = Generator(generator_model_name, cache_dir, device)
        self.reader = Reader(reader_model_name, cache_dir, device)

    def __call__(self, input_data):
        results = self.generator(input_data)
        results = self.reader(results)

        return results
