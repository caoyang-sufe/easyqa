# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

import os
import logging
from src.base import BaseClass

class BaseDataset(BaseClass):
	dataset_name = None
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
	dataset_name = "Extractive"
	def __init__(self, data_dir, **kwargs):
		super(BaseExtractiveDataset, self).__init__(data_dir, **kwargs)


class BaseGenerativeDataset(BaseDataset):
	dataset_name = "Generative"
	def __init__(self, data_dir, **kwargs):
		super(BaseGenerativeDataset, self).__init__(data_dir, **kwargs)


class BaseMultipleChoiceDataset(BaseDataset):
	dataset_name = "Multiple-choice"
	batch_data_keys = ["article_id",
					   "question_id",
					   "article",
					   "question",
					   "options",
					   "answer",
					   ]
	def __init__(self, data_dir, **kwargs):
		super(BaseMultipleChoiceDataset, self).__init__(data_dir, **kwargs)

	# Generate model inputs for multiple-choice MRC models
	# @param batch
	def generate_model_inputs(cls,
							  batch,
							  tokenizer,
							  model_name,
							  **kwargs,
							  ):
		if model_name == "LIAMF-USP/roberta-large-finetuned-race":
			max_length = kwargs["max_length"]
			batch_inputs = list()
			for data in batch:
				article = data["article"]
				question = data["question"]
				option = data["options"]
				flag = question.find('_') == -1
				inputs = list()
				for choice in option:
					question_choice = question + ' ' + choice if flag else question.replace('_', choice)
					input_ = tokenizer(article,
									   question_choice,
									   add_special_tokens = True,
									   max_length = max_length,
									   padding = "max_length",
									   truncation = True,
									   return_overflowing_tokens = False,
									   return_tensors = "pt",
									   ) # (1, max_length)
					inputs.append(input_)
				batch_inputs.append(inputs)
			input_ids = torch.cat([torch.cat([input_["input_ids"] for input_ in inputs]).unsqueeze(0) for inputs in batch_inputs]) # (batch_size, n_option, max_length)
			attention_mask = torch.cat([torch.cat([input_["attention_mask"] for input_ in inputs]).unsqueeze(0) for inputs in batch_inputs]) # (batch_size, n_option, max_length)
			model_inputs = {"input_ids": input_ids,
							"attention_mask": attention_mask,
							}
		elif model_name == "potsawee/longformer-large-4096-answering-race":
			max_length = kwargs["max_length"]
			batch_inputs = list()
			for data in batch:
				article = data["article"]
				question = data["question"]
				option = data["options"]
				article_question = [f"{question} {tokenizer.bos_token} article"] * 4
				inputs = tokenizer(article_question,
								   option,
								   max_length = max_length,
								   padding = "longest",
								   truncation = True,
								   return_tensors = "pt",
								   ) # (4, max_length)
				batch_inputs.append(inputs)
			input_ids = torch.cat([inputs["input_ids"].unsqueeze(0) for inputs in batch_inputs], axis=0)
			attention_mask = torch.cat([inputs["attention_mask"].unsqueeze(0) for inputs in batch_inputs], axis=0)
			model_inputs = {"input_ids": input_ids,
							"attention_mask": attention_mask,
							}
		else:
			raise NotImplementedError(model_name)
		return model_inputs

	# Check data keys in yield batch
	# @param batch: @yield of function `yield_batch`
	def check_batch(self, batch):
		for key in self.batch_data_keys:
			assert key in data, f"{key} not found in yield batch"
