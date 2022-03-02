# adapted from https://github.com/patil-suraj/question_generation which is under the MIT License

import itertools
import logging
from typing import Optional, Dict, Union
import os
from nltk import sent_tokenize

import torch
from transformers import(
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)

class QGPipeline:
    """Poor man's QG pipeline"""
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        ans_model: PreTrainedModel,
        ans_tokenizer: PreTrainedTokenizer,
        qg_format: str,
        device: str,
        default_answers = None,
    ):
        self.model = model
        self.tokenizer = tokenizer

        self.ans_model = ans_model
        self.ans_tokenizer = ans_tokenizer

        self.qg_format = qg_format
        self.default_answers = default_answers
        self.device = device
        if self.model.device != self.device:
            self.model.to(self.device).eval()
            if device == "cpu":
                self.model = torch.quantization.quantize_dynamic(self.model.float(), {torch.nn.Linear}, dtype=torch.qint8)
            else:  
                self.model = self.model.half().to(device)

        if self.ans_model is not self.model:
            if self.ans_model.device != self.device:
                self.ans_model.to(self.device).eval()
                if device == "cpu":
                    self.ans_model = torch.quantization.quantize_dynamic(self.ans_model.float(), {torch.nn.Linear}, dtype=torch.qint8)
                else:  
                    self.ans_model = self.ans_model.half().to(device)

        assert self.model.__class__.__name__ in ["T5ForConditionalGeneration", "BartForConditionalGeneration"]
        
        if "T5ForConditionalGeneration" in self.model.__class__.__name__:
            self.model_type = "t5"
        else:
            self.model_type = "bart"

    def __call__(self, inputs: str, **generate_kwargs):
        self.model.eval()
        self.ans_model.eval()
        ret = []
        with torch.no_grad():
          
          if type(inputs) is str:
            inputs = [inputs]
          default_answers=[]
          if 'default_answers' in generate_kwargs:
            default_answers = generate_kwargs['default_answers']
            if type(default_answers[0]) is str:
              default_answers = [default_answers] * len(inputs)
          if len(default_answers) < len(inputs):
            default_answers.extend([[]]*(len(inputs)-len(default_answers)))
          #TODO - we could do in batches that is approximately N words to maximize GPU usage
          for input, default_answer in zip(inputs, default_answers):
            qg_examples = []
            input = " ".join(input.split())
            sents, answers = self._extract_answers(input)
            if self.default_answers:
              answers.append(self.default_answers)
            if default_answer:
              answers.append(default_answer)
            flat_answers = list(itertools.chain(*answers))
            
            if len(flat_answers) == 0:
              ret.append([])
              continue
            answers = [flat_answers]*len(sents) # multi-way q/a
            if self.qg_format == "prepend":
                qg_examples.extend(self._prepare_inputs_for_qg_from_answers_prepend(inputs, answers))
            else:
                qg_examples.extend(self._prepare_inputs_for_qg_from_answers_hl(sents, answers))
            if  qg_examples:
              qg_inputs = [example['source_text'] for example in qg_examples]
              questions = self._generate_questions(qg_inputs)
              output = list(set([(example['answer'], que) for example, que in zip(qg_examples, questions)]))
              ret.append([{'answer': answer, 'question': que} for answer, que in output])
            else:
              ret.append([])
        return ret
    
    def _generate_questions(self, inputs):
        inputs = self._tokenize(inputs, padding=True, truncation=True)
        
        outs = self.model.generate(
            input_ids=inputs['input_ids'].to(self.device), 
            attention_mask=inputs['attention_mask'].to(self.device), 
            max_length=32,
            num_beams=4,
        )
        
        questions = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        return questions
    
    def _extract_answers(self, context):
        sents, inputs = self._prepare_inputs_for_ans_extraction(context)
        inputs = self._tokenize(inputs, padding=True, truncation=True)
        self.ans_model.eval()
        with torch.no_grad():
          outs = self.ans_model.generate(
              input_ids=inputs['input_ids'].to(self.device), 
              attention_mask=inputs['attention_mask'].to(self.device), 
              max_length=32,
          )
        
        dec = [self.ans_tokenizer.decode(ids, skip_special_tokens=False) for ids in outs]
        answers = [item.replace("<pad>","").replace("  ", " ").strip().split('<sep>') for item in dec]
        answers = [i[:-1] for i in answers if i !=[]]
        
        return sents, answers
    
    def _tokenize(self,
        inputs,
        padding=True,
        truncation=True,
        add_special_tokens=True,
        max_length=512
    ):
        inputs = self.tokenizer.batch_encode_plus(
            inputs, 
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding="max_length" if padding else False,
            pad_to_max_length=padding,
            return_tensors="pt"
        )
        return inputs
    
    def _prepare_inputs_for_ans_extraction(self, text):
        sents = sent_tokenize(text)

        inputs = []
        for i in range(len(sents)):
            source_text = "extract answers:"
            for j, sent in enumerate(sents):
                if i == j:
                    sent = "<hl> %s <hl>" % sent
                source_text = "%s %s" % (source_text, sent)
                source_text = source_text.strip()
            
            if self.model_type == "t5":
                source_text = source_text + " </s>"
            inputs.append(source_text)

        return sents, inputs
    
    def _prepare_inputs_for_qg_from_answers_hl(self, sents, answers):
        inputs = []
        for i, answer in enumerate(answers):
            if len(answer) == 0: continue
            for answer_text in answer:
                sent = sents[i]
                sents_copy = sents[:]
                
                answer_text = answer_text.strip()
                if answer_text.lower() not in sent.lower(): continue
                ans_start_idx = sent.lower().index(answer_text.lower())
                
                sent = f"{sent[:ans_start_idx]} <hl> {answer_text} <hl> {sent[ans_start_idx + len(answer_text): ]}"
                sents_copy[i] = sent
                
                source_text = " ".join(sents_copy)
                source_text = f"generate question: {source_text}" 
                if self.model_type == "t5":
                    source_text = source_text + " </s>"
                
                inputs.append({"answer": answer_text, "source_text": source_text})
        
        return inputs
    
    def _prepare_inputs_for_qg_from_answers_prepend(self, context, answers):
        flat_answers = list(itertools.chain(*answers))
        examples = []
        for answer in flat_answers:
            source_text = f"answer: {answer} context: {context}"
            if self.model_type == "t5":
                source_text = source_text + " </s>"
            
            examples.append({"answer": answer, "source_text": source_text})
        return examples

    
class MultiTaskQAQGPipeline(QGPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self, inputs: Union[Dict, str], **generate_kwargs):
        if type(inputs) in (list, str):
            # do qg
            return super().__call__(inputs, **generate_kwargs)
        else:
            # do qa
            return self._extract_answer(inputs["question"], inputs["context"], **generate_kwargs)
    
    def _prepare_inputs_for_qa(self, question, context):
        source_text = f"question: {question}  context: {context}"
        if self.model_type == "t5":
            source_text = source_text + " </s>"
        return  source_text
    
    def _extract_answer(self, question, context):
        source_text = self._prepare_inputs_for_qa(question, context)
        inputs = self._tokenize([source_text], padding=False)
    
        outs = self.model.generate(
            input_ids=inputs['input_ids'].to(self.device), 
            attention_mask=inputs['attention_mask'].to(self.device), 
            max_length=16,
        )

        answer = self.tokenizer.decode(outs[0], skip_special_tokens=True)
        return answer


class E2EQGPipeline:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str,
    ) :

        self.model = model
        self.tokenizer = tokenizer

        self.device = device
        self.model.to(self.device)

        assert self.model.__class__.__name__ in ["T5ForConditionalGeneration", "BartForConditionalGeneration"]
        
        if "T5ForConditionalGeneration" in self.model.__class__.__name__:
            self.model_type = "t5"
        else:
            self.model_type = "bart"
        
        self.default_generate_kwargs = {
            "max_length": 256,
            "num_beams": 4,
            "length_penalty": 1.5,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
        }
    
    def __call__(self, context: str, **generate_kwargs):
        inputs = self._prepare_inputs_for_e2e_qg(context)

        # TODO: when overrding default_generate_kwargs all other arguments need to be passsed
        # find a better way to do this
        if not generate_kwargs:
            generate_kwargs = self.default_generate_kwargs
        
        input_length = inputs["input_ids"].shape[-1]
        
        # max_length = generate_kwargs.get("max_length", 256)
        # if input_length < max_length:
        #     logger.warning(
        #         "Your max_length is set to {}, but you input_length is only {}. You might consider decreasing max_length manually, e.g. summarizer('...', max_length=50)".format(
        #             max_length, input_length
        #         )
        #     )

        outs = self.model.generate(
            input_ids=inputs['input_ids'].to(self.device), 
            attention_mask=inputs['attention_mask'].to(self.device),
            **generate_kwargs
        )

        prediction = self.tokenizer.decode(outs[0], skip_special_tokens=True)
        questions = prediction.split("<sep>")
        questions = [question.strip() for question in questions[:-1]]
        return questions
    
    def _prepare_inputs_for_e2e_qg(self, context):
        source_text = f"generate questions: {context}"
        if self.model_type == "t5":
            source_text = source_text + " </s>"
        
        inputs = self._tokenize([source_text], padding=False)
        return inputs
    
    def _tokenize(
        self,
        inputs,
        padding=True,
        truncation=True,
        add_special_tokens=True,
        max_length=512
    ):
        inputs = self.tokenizer.batch_encode_plus(
            inputs, 
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding="max_length" if padding else False,
            pad_to_max_length=padding,
            return_tensors="pt"
        )
        return inputs


SUPPORTED_TASKS = {
    "question-generation": {
        "impl": QGPipeline,
        "default": {
            "model": "valhalla/t5-small-qg-hl" if 'PII' not in os.getcwd() else os.path.join(os.getcwd(),'../t5-small-qa-qg-hl/'),
            "ans_model": "valhalla/t5-small-qa-qg-hl" if 'PII' not in os.getcwd() else os.path.join(os.getcwd(),'../t5-small-qa-qg-hl/'),
        }
    },
    "multitask-qa-qg": {
        "impl": MultiTaskQAQGPipeline,
        "default": {
            "model": "valhalla/t5-small-qa-qg-hl" if 'PII' not in os.getcwd() else os.path.join(os.getcwd(),'../t5-small-qa-qg-hl/'),
        }
    },
    "e2e-qg": {
        "impl": E2EQGPipeline,
        "default": {
            "model": "valhalla/t5-small-e2e-qg" if 'PII' not in os.getcwd() else os.path.join(os.getcwd(),'../t5-small-e2e-qg/'),
        }
    }
}

def pipeline(
    task: str,
    model: Optional = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    qg_format: Optional[str] = "highlight",
    ans_model: Optional = None,
    ans_tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    device: str = "cpu",
    **kwargs,
):

    # Retrieve the task
    if task not in SUPPORTED_TASKS:
        raise KeyError("Unknown task {}, available tasks are {}".format(task, list(SUPPORTED_TASKS.keys())))

    targeted_task = SUPPORTED_TASKS[task]
    task_class = targeted_task["impl"]
    models_same=False
    # Use default model/config/tokenizer for the task if no model is provided
    if model is None:
        model = targeted_task["default"]["model"]
    if ans_model is None:
        ans_model = targeted_task["default"].get("ans_model", None)
    if isinstance(model, str) and isinstance(ans_model, str) and model == ans_model:
      models_same = True
    # Try to infer tokenizer from model or config name (if provided as str)
    if tokenizer is None:
        if isinstance(model, str):
            tokenizer = model
        else:
            # Impossible to guest what is the right tokenizer here
            raise Exception(
                "Impossible to guess which tokenizer to use. "
                "Please provided a PretrainedTokenizer class or a path/identifier to a pretrained tokenizer."
            )
    
    # Instantiate tokenizer if needed
    if isinstance(tokenizer, (str, tuple)):
        if isinstance(tokenizer, tuple):
            # For tuple we have (tokenizer name, {kwargs})
            tokenizer = AutoTokenizer.from_pretrained(tokenizer[0], **tokenizer[1])
        else:
            print(tokenizer)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)

                
    
    # Instantiate model if needed
    if isinstance(model, str):
        model = AutoModelForSeq2SeqLM.from_pretrained(model).eval()
        if device == "cpu":
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        else:
            model = model.half().to(device)
                
    if task == "question-generation":
        if ans_model is None:
            # load default ans model
            ans_model = targeted_task["default"]["ans_model"]
            ans_tokenizer = AutoTokenizer.from_pretrained(ans_model)
            if models_same:
              ans_model = model
            else:
              ans_model = AutoModelForSeq2SeqLM.from_pretrained(ans_model).eval()
              if device == "cpu":
                ans_model = torch.quantization.quantize_dynamic(ans_model, {torch.nn.Linear}, dtype=torch.qint8)
              else:
                ans_model = ans_model.half().to(device)
                
        else:
            # Try to infer tokenizer from model or config name (if provided as str)
            if models_same:
              ans_tokenizer = tokenizer
            elif ans_tokenizer is None:
                if isinstance(ans_model, str):
                    ans_tokenizer = ans_model
                else:
                    # Impossible to guest what is the right tokenizer here
                    raise Exception(
                        "Impossible to guess which tokenizer to use. "
                        "Please provided a PretrainedTokenizer class or a path/identifier to a pretrained tokenizer."
                    )
            
            # Instantiate tokenizer if needed
            if isinstance(ans_tokenizer, (str, tuple)):
                if isinstance(ans_tokenizer, tuple):
                    # For tuple we have (tokenizer name, {kwargs})
                    ans_tokenizer = AutoTokenizer.from_pretrained(ans_tokenizer[0], **ans_tokenizer[1])
                else:
                    ans_tokenizer = AutoTokenizer.from_pretrained(ans_tokenizer)

            if models_same:
              ans_model = model
            elif isinstance(ans_model, str):
                ans_model = AutoModelForSeq2SeqLM.from_pretrained(ans_model).eval()
                if device == "cpu":
                    ans_model = torch.quantization.quantize_dynamic(ans_model, {torch.nn.Linear}, dtype=torch.qint8)
                else:
                    ans_model = ans_model.half().to(device)
    
    if task == "e2e-qg":
        return task_class(model=model, tokenizer=tokenizer, device=device)
    elif task == "question-generation":
        return task_class(model=model, tokenizer=tokenizer, ans_model=ans_model, ans_tokenizer=ans_tokenizer, qg_format=qg_format, device=device)
    else:
        return task_class(model=model, tokenizer=tokenizer, ans_model=model, ans_tokenizer=tokenizer, qg_format=qg_format, device=device)
