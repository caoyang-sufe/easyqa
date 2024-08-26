# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

import logging
from src.base import BaseClass

class BaseDataset(BaseClass):

	checked_data_dirs = list()
	
	def __init__(self, **kwargs):
		super(BaseDataset, self).__init__(**kwargs)

	@classmethod
	def generate_model_inputs(cls, batch, tokenizer, **kwargs):
		raise NotImplementedError()

	# Generator to yield batch data
	def yield_batch(self, **kwargs):
		raise NotImplementedError()

	# Check files and directories of datasets
	def check_data_dir(self, data_dir):
		logging.info()
		if self.checked_data_dirs:
			
			logging.info

		else:
			logging.info(f"- Nothing to check in {}")
		

class BaseExtractiveDataset(BaseDataset):

	def __init__(self, **kwargs):
		super(BaseExtractiveDataset, self).__init__(**kwargs)


class BaseGenerativeDataset(BaseDataset):

	def __init__(self, **kwargs):
		super(BaseGenerativeDataset, self).__init__(**kwargs)


class BaseMultipleChoiceDataset(BaseDataset):

	def __init__(self, **kwargs):
		super(BaseMultipleChoiceDataset, self).__init__(**kwargs)