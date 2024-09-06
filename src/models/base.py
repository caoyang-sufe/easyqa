# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

import torch
import logging

from src.base import BaseClass
from src.datasets import (ExtractiveDataset,
						  GenerativeDataset,
						  MultipleChoiceDataset,
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

	def run(self, batch, **kwargs):
		model_inputs = self.generate_model_inputs(batch, **kwargs)
		model_outputs = self.model(**model_inputs)
		batch_start_logits = model_outputs.start_logits
		batch_end_logits = model_outputs.end_logits
		del model_inputs, model_outputs
		batch_size = model_inputs.size(0)
		batch_predicts = list()
		for i in range(batch_size):
			start_index = batch_start_logits[i].argmax().item()
			end_index = batch_end_logits[i].argmax().item()
			input_ids = model_inputs["input_ids"][i]
			batch_predicts.append(input_ids[start_index, end_index + 1])
		return batch_start_logits, batch_end_logits, batch_predicts
		

	def generate_model_inputs(self, batch, **kwargs):
		return ExtractiveDataset.generate_model_inputs(
			batch = batch,
			tokenizer = self.tokenizer,
			model_name = self.model_name,
			**kwargs,
		)

class GenerativeModel(BaseModel):

	def __init__(self, model_path, device, **kwargs):
		super(GenerativeModel, self).__init__(model_path, device, **kwargs)


class MultipleChoiceModel(BaseModel):

	def __init__(self, model_path, device, **kwargs):
		super(MultipleChoiceModel, self).__init__(model_path, device, **kwargs)

	# @param data: Dict[article(List[Str]), question(List[Str]), options(List[List[Str]])]
	# @return batch_logits: FloatTensor(batch_size, n_option)
	# @return batch_predicts: List[Str] (batch_size, )
	def run(self, batch, **kwargs):
		model_inputs = self.generate_model_inputs(batch, **kwargs)
		model_outputs = self.model(**model_inputs)
		batch_logits = model_outputs.logits
		del model_inputs, model_outputs
		batch_predicts = ["ABCD"[torch.argmax(logits).item()] for logits in batch_logits]
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
		