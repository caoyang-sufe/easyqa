# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

import os
import logging
from src.base import BaseClass

class BaseDataset(BaseClass):

	checked_data_dirs = []
	
	def __init__(self, data_dir, **kwargs):
		super(BaseDataset, self).__init__(**kwargs)
		self.data_dir = data_dir
		self.check_data_dir()

	@classmethod
	def generate_model_inputs(cls, batch, tokenizer, **kwargs):
		raise NotImplementedError()

	# Generator to yield batch data
	def yield_batch(self, **kwargs):
		raise NotImplementedError()

	# Check files and directories of datasets
	def check_data_dir(self):
		logging.info(f"Check data directory: {self.data_dir}")
		if self.checked_data_dirs:
			for checked_data_dir in self.checked_data_dirs:
				if os.path.exists(os.path.join(self.data_dir, checked_data_dir)):
					logging.info(f"√ {checked_data_dir}")
				else:
					logging.warning(f"× {checked_data_dir}")
		else:
			logging.warning("- Nothing to check!")
		

class BaseExtractiveDataset(BaseDataset):

	def __init__(self, data_dir, **kwargs):
		super(BaseExtractiveDataset, self).__init__(data_dir, **kwargs)


class BaseGenerativeDataset(BaseDataset):

	def __init__(self, data_dir, **kwargs):
		super(BaseGenerativeDataset, self).__init__(data_dir, **kwargs)


class BaseMultipleChoiceDataset(BaseDataset):

	def __init__(self, data_dir, **kwargs):
		super(BaseMultipleChoiceDataset, self).__init__(data_dir, **kwargs)