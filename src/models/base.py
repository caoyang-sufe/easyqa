# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

import logging

from src.base import BaseClass
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



