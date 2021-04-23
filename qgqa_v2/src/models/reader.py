# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2020/10/12
@Desc               :
@Last modified by   : Bao
@Last modified date : 2021/1/17
"""

from tqdm import tqdm
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, T5ForConditionalGeneration


class Reader:
    """
    Examples:
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
            'allenai/unifiedqa-t5-base',
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
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir if cache_dir else None,
        )
        self.model.to(device)
        self.model.eval()

        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def __call__(self, input_data, max_length=None, batch_size=8, beam_size=1):
        num_questions = len(input_data[0]['questions'])

        _input_data = []
        for entry in input_data:
            for question in entry['questions']:
                _input_data.append({'context': entry['context'], 'question': question, 'answer': entry['answer']})
        all_scores = []
        num_batches = (len(_input_data) + batch_size - 1) // batch_size
        for step in tqdm(range(num_batches), desc='Generating answers'):
            batch_start = step * batch_size
            batch_end = min((step + 1) * batch_size, len(_input_data))
            batch_text = []
            for entry in _input_data[batch_start:batch_end]:
                batch_text.append('{} \\n {}'.format(entry['question'], entry['context']))
            inputs = self.tokenizer.batch_encode_plus(
                batch_text,
                padding='max_length',
                truncation='longest_first',
                max_length=max_length,
                return_tensors='pt',
            )

            for key, value in inputs.items():
                inputs[key] = value.to(self.model.device)

            ids_with_beam = self.model.generate(num_beams=beam_size, num_return_sequences=beam_size, **inputs)
            ids_with_beam = ids_with_beam.reshape([batch_end - batch_start, beam_size, -1])
            for ids, entry in zip(ids_with_beam, _input_data[batch_start:batch_end]):
                hyp = entry['answer'][0]
                refs = self.tokenizer.batch_decode(ids, skip_special_tokens=True)
                all_scores.append(max([self.scorer.score(ref, hyp)['rougeL'].fmeasure for ref in refs]))

        for i in range(len(input_data)):
            start = i * num_questions
            end = (i + 1) * num_questions
            input_data[i]['scores_q'] = all_scores[start:end]

        return input_data
