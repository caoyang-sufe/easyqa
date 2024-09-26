# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

import torch
import string
import logging

from src.base import BaseClass
from src.datasets import (ExtractiveDataset,
						  GenerativeDataset,
						  MultipleChoiceDataset,
						  RaceDataset,
						  DreamDataset,
						  SquadDataset,
						  HotpotqaDataset,
						  MusiqueDataset,
						  TriviaqaDataset
						  )
from transformers import AutoTokenizer, AutoModel

class BaseModel(BaseClass):
	Tokenizer = AutoTokenizer
	Model = AutoModel

	def __init__(self, model_path, device, **kwargs):
		super(BaseModel, self).__init__(**kwargs)
		self.model_path = model_path
		self.device = device
		# Load model and tokenizer
		self.load_tokenizer()
		self.load_vocab()
		# self.load_model()

	# Load tokenizer
	def load_tokenizer(self):
		self.tokenizer = self.Tokenizer.from_pretrained(self.model_path)

	# Load pretrained model
	def load_model(self):
		self.model = self.Model.from_pretrained(self.model_path).to(self.device)

	# Load vocabulary (in format of Dict[id: token])
	def load_vocab(self):
		self.vocab = {token_id: token for token, token_id in self.tokenizer.get_vocab().items()}


class ExtractiveModel(BaseModel):

	def __init__(self, model_path, device, **kwargs):
		super(ExtractiveModel, self).__init__(model_path, device, **kwargs)

	# @param batch: @yield in function `yield_batch` of Dataset object
	# @return batch_start_logits: FloatTensor(batch_size, max_length)
	# @return batch_end_logits: FloatTensor(batch_size, max_length)
	# @return batch_predicts: List[Str] with length batch_size
	# Default for most extractive question-answering models
	def forward(self, batch, **kwargs):
		model_inputs = self.generate_model_inputs(batch, **kwargs)
		for key in model_inputs:
			model_inputs[key] = model_inputs[key].to(self.device)
		model_outputs = self.model(**model_inputs)
		# 2024/09/13 11:08:21
		# Note: Skip the first token <s> or [CLS] in most situation
		batch_start_logits = model_outputs.start_logits[:, 1:]
		batch_end_logits = model_outputs.end_logits[:, 1:]
		batch_input_ids = model_inputs["input_ids"][:, 1:]
		del model_inputs, model_outputs
		batch_size = batch_start_logits.size(0)
		batch_predicts = list()
		batch_input_tokens = list()
		for i in range(batch_size):
			start_index = batch_start_logits[i].argmax().item()
			end_index = batch_end_logits[i].argmax().item()
			input_ids = batch_input_ids[i]
			input_tokens = list(map(lambda _token_id: self.vocab[_token_id.item()], input_ids))
			predict_tokens = list()
			for index in range(start_index, end_index + 1):
				predict_tokens.append((index, self.vocab[input_ids[index].item()]))
				# predict_tokens.append(self.vocab[input_ids[index].item()])
			batch_predicts.append(predict_tokens)
			batch_input_tokens.append(input_tokens)
		return batch_start_logits, batch_end_logits, batch_predicts, batch_input_tokens

	# Generate model inputs
	# @param batch: @yield in function `yield_batch` of Dataset object
	def generate_model_inputs(self, batch, **kwargs):
		return ExtractiveDataset.generate_model_inputs(
			batch = batch,
			tokenizer = self.tokenizer,
			model_name = self.model_name,
			**kwargs,
		)

	# Use question-answering pipeline provided by transformers
	# See `QuestionAnsweringPipeline.preprocess` in ./site-packages/transformers/pipelines/question_answering.py for details
	# @param context: Str / List[Str] (batch)
	# @param question: Str / List[Str] (batch)
	# @return pipeline_outputs: Dict[score: Float, start: Int, end: Int, answer: Str]
	def easy_pipeline(self, context, question):
		# context = """Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny\'s Child. Managed by her father, Mathew Knowles, the group became one of the world\'s best-selling girl groups of all time. Their hiatus saw the release of Beyoncé\'s debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy"."""
		# question = """When did Beyonce start becoming popular?"""
		pipeline_inputs = {"context": context, "question": question}
		question_answering_pipeline = pipeline(task = "question-answering",
											   model = self.model,
											   tokenizer = tokenizer,
											   )
		pipeline_outputs = question_answering_pipeline(pipeline_inputs)
		return pipeline_outputs

class GenerativeModel(BaseModel):

	def __init__(self, model_path, device, **kwargs):
		super(GenerativeModel, self).__init__(model_path, device, **kwargs)

	# @param batch: @yield in function `yield_batch` of Dataset object
	# @return batch_start_logits: FloatTensor(batch_size, max_length)
	# @return batch_end_logits: FloatTensor(batch_size, max_length)
	# @return batch_predicts: List[Str] with length batch_size
	# 2024/09/26 10:53:36 Default for Chatglm series only
	def forward(self, batch, **kwargs):
		model_inputs = self.generate_model_inputs(batch, **kwargs)
		for key in model_inputs:
			model_inputs[key] = model_inputs[key].to(self.device)
		model_outputs = self.model(**model_inputs)
		batch_loss = model_outputs.loss	# Note: None in `model.eval()` settings
		batch_logits = model_outputs.logits	# (batch_size, max_length, vocab_size)
		batch_past_key_value = model_outputs.past_key_values	# List[Tuple(key: tensor, value: tensor)], the length of List is 28 usually
		del model_inputs, model_outputs
		return batch_loss, batch_logits, batch_past_key_value

	# Generate model inputs
	# @param batch: @yield in function `yield_batch` of Dataset object
	def generate_model_inputs(self, batch, **kwargs):
		return GenerativeDataset.generate_model_inputs(
			batch = batch,
			tokenizer = self.tokenizer,
			model_name = self.model_name,
			**kwargs,
		)

	# Use ??? pipeline provided by transformers
	# @param context: Str / List[Str] (batch)
	# @param question: Str / List[Str] (batch)
	def easy_pipeline(self, context, question):
		# context = """Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny\'s Child. Managed by her father, Mathew Knowles, the group became one of the world\'s best-selling girl groups of all time. Their hiatus saw the release of Beyoncé\'s debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy"."""
		# question = """When did Beyonce start becoming popular?"""
		raise NotImplementedError()


class MultipleChoiceModel(BaseModel):

	def __init__(self, model_path, device, **kwargs):
		super(MultipleChoiceModel, self).__init__(model_path, device, **kwargs)

	# @param data: Dict[article(List[Str]), question(List[Str]), options(List[List[Str]])]
	# @return batch_logits: FloatTensor(batch_size, n_option)
	# @return batch_predicts: List[Str] (batch_size, )
	# Default for most multiple-choice question-answering models
	def forward(self, batch, **kwargs):
		model_inputs = self.generate_model_inputs(batch, **kwargs)
		for key in model_inputs:
			model_inputs[key] = model_inputs[key].to(self.device)
		model_outputs = self.model(**model_inputs)
		batch_logits = model_outputs.logits
		del model_inputs, model_outputs
		batch_predicts = [torch.argmax(logits).item() for logits in batch_logits]
		return batch_logits, batch_predicts

	# Generate model inputs
	# @param batch: @yield in function `yield_batch` of Dataset object
	# @param max_length: Max length of input tokens
	def generate_model_inputs(self, batch, **kwargs):
		return MultipleChoiceDataset.generate_model_inputs(
			batch = batch,
			tokenizer = self.tokenizer,
			model_name = self.model_name,
			**kwargs,
		)
		