# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2020/10/13
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/10/20
"""

from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelWithLMHead


class Generator:
    """

    Example:
        input_data = [
            {'context': 'My name is <hl> Sarah <hl>.'},
            {'context': 'My name is Sarah and I live in <hl> London <hl>.'},
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
        self.loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')

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

        outputs = self.model(input_ids=inputs['input_ids'], labels=inputs['input_ids'])
        question_ids = self.model.generate(**inputs)
        logits = outputs[1]
        loss = self.loss_fct(logits.view(-1, logits.size(-1)), inputs['input_ids'].view(-1)).detach().cpu().tolist()

        for i, (ids, log_ppl) in enumerate(zip(question_ids, loss)):
            input_data[i]['question'] = self.tokenizer.decode(ids, skip_special_tokens=True)
            input_data[i]['log_ppl'] = log_ppl

        return input_data
