# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2020/10/13
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/10/13
"""

from src.models import Pipeline


def main():
    import json
    import torch

    input_data = [
        {'context': 'My name is <hl> Sarah <hl> .', 'raw_answer': 'Sarah'},
        {'context': 'My name is Sarah and I live in <hl> London <hl> .', 'raw_answer': 'London'},
        {'context': 'Sarah lived in London. Jone lived in <hl> Canada <hl> .', 'raw_answer': 'Canada'},
        {'context': 'Sarah <hl> lived <hl> in London. Jone lived in Canada.', 'raw_answer': 'lived'},
        {'context': 'Jone election and policies have sparked <hl> numerous protests <hl> .'},
        {'context': 'Jone bought the Miss Universe brand of beauty pageants in <hl> 1996 <hl> , and sold it in 2015.'},
    ]
    pipeline = Pipeline(
        generator_model_name='valhalla/t5-base-qg-hl',
        reader_model_name='deepset/bert-base-cased-squad2',
        cache_dir='/home/qbbao/003_downloads/cache_transformers-3.1.0',
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    results = pipeline(input_data)
    print(json.dumps(results, ensure_ascii=False, indent=4))


if __name__ == '__main__':
    main()
