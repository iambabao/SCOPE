# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2020/10/13
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/10/13
"""

from transformers import AutoTokenizer, AutoModelWithLMHead


class Generator:
    """

    Example:
        input_data = [
            {'context': 'My name is <hl> Sarah <hl> .'},
            {'context': 'My name is Sarah and I live in <hl> London <hl> .'},
            {'context': 'Sarah lived in London. Jone lived in <hl> Canada <hl>.'},
        ]
        generator = Generator(
            'valhalla/t5-base-qg-hl',
            'your_cache_dir',
            'cuda' if torch.cuda.is_available() else 'cpu',
        )

        results = generator(input_data)
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

    def __call__(self, input_data, max_length=None):
        batch_text = [entry['context'] for entry in input_data]
        inputs = self.tokenizer.batch_encode_plus(
            batch_text,
            padding='max_length',
            truncation='longest_first',
            max_length=max_length,
            return_tensors='pt',
        )
        for key, value in inputs.items():
            inputs[key] = value.to(self.model.device)

        question_ids = self.model.generate(**inputs)

        for i, ids in enumerate(question_ids):
            input_data[i]['question'] = self.tokenizer.decode(ids, skip_special_tokens=True)

        return input_data
