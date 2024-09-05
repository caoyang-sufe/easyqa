# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

import torch
import logging

from src.base import BaseClass
from src.datasets import (BaseExtractiveDataset,
						  BaseGenerativeDataset,
						  BaseMultipleChoiceDataset,
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
		# self.model = self.load_model()
		self.tokenizer = self.load_tokenizer()

	def load_tokenizer(self):
		tokenizer = self.Tokenizer.from_pretrained(self.model_path)
		return tokenizer

	def load_model(self):
		model = self.Model.from_pretrained(self.model_path).to(self.device)
		return model


class BaseExtractiveModel(BaseModel):

	def __init__(self, model_path, device, **kwargs):
		super(BaseExtractiveModel, self).__init__(model_path, device, **kwargs)


class BaseGenerativeModel(BaseModel):

	def __init__(self, model_path, device, **kwargs):
		super(BaseGenerativeModel, self).__init__(model_path, device, **kwargs)


class BaseMultipleChoiceModel(BaseModel):

	def __init__(self, model_path, device, **kwargs):
		super(BaseMultipleChoiceModel, self).__init__(model_path, device, **kwargs)

	# @param data: Dict[article(List[Str]), question(List[Str]), options(List[List[Str]])]
	# @return batch_logits: FloatTensor(batch_size, n_option)
	# @return batch_predicts: List[Str] (batch_size, )
	def run(self, batch, max_length=512):
		model_inputs = self.generate_model_inputs(batch, max_length=max_length)
		batch_logits = self.model(**model_inputs).logits
		del model_inputs
		batch_predicts = ["ABCD"[torch.argmax(logits).item()] for logits in batch_logits]
		return batch_logits, batch_predicts

	# Generate model inputs
	# @param batch: @yield in function `yield_batch` of Dataset object
	# @param max_length: Max length of input tokens
	def generate_model_inputs(self, batch, max_length):
		return BaseMultipleChoiceDataset.generate_model_inputs(
			batch = batch,
			tokenizer = self.tokenizer,
			model_name = self.model_name,
			max_length = max_length,
		)
		