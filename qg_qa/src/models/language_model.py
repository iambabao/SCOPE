# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2020/10/14
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/10/14
"""


import torch
from transformers import AutoTokenizer, AutoModelWithLMHead


class LanguageModel:
    """

    Example:
        input_data = [
            {'context': 'My name is Sarah.'},
            {'context': 'My name is Sarah and I live in London.'},
            {'context': 'Sarah lived in London. Jone lived in Canada.'},
        ]
        lm = LanguageModel(
            'gpt2',
            'your_cache_dir',
            'cuda' if torch.cuda.is_available() else 'cpu',
        )

        results = lm(input_data, lm_key='context')
        print(json.dumps(results, ensure_ascii=False, indent=4))
    """

    def __init__(self, model_name_or_path, cache_dir=None, device='cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir if cache_dir else None,
        )
        self.model = AutoModelWithLMHead.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir if cache_dir else None,
        )
        self.model.to(device)
        self.model.eval()

    def __call__(self, input_data, lm_key, max_length=None):
        for i, entry in enumerate(input_data):
            inputs = self.tokenizer.encode_plus(entry[lm_key], return_tensors='pt')
            for key, value in inputs.items():
                inputs[key] = value.to(self.model.device)

            outputs = self.model(**inputs, labels=inputs['input_ids'])
            loss = outputs[0]
            # perplexity = exp\left(-\frac{1}{|Y|} \sum_{i=1}^{i=|Y|}{P_{LM}\left(y_{i}|y_{0:i-1}\right)}\right)
            perplexity = torch.exp(loss).detach().cpu().tolist()
            # score = exp\left(\frac{1}{|Y|} \sum_{i=1}^{i=|Y|}{P_{LM}\left(y_{i}|y_{0:i-1}\right)}\right)
            score = torch.exp(-1.0 * loss).detach().cpu().tolist()

            input_data[i]['perplexity'] = perplexity  # lower is better
            input_data[i]['score'] = score  # higher is better

        return input_data
