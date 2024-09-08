# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

import os
import torch
import logging

from src.base import BaseClass

class BaseDataset(BaseClass):
	dataset_name = None
	checked_data_dirs = []
	batch_data_keys = []
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

	# Check data keys in yield batch
	# @param batch: @yield of function `yield_batch`
	def check_batch_data_keys(self, batch):
		for key in self.batch_data_keys:
			assert key in batch[0], f"{key} not found in yield batch"


class ExtractiveDataset(BaseDataset):
	dataset_name = "Extractive"
	batch_data_keys = ["context",	# List[Tuple[Str, List[Str]]], i.e. List of [title, article[sentence]]
					   "question",	# Str
					   "answers",	# List[Str]
					   "answer_starts",	# List[Int]
					   "answer_ends",	# List[Int]
					   ]
	def __init__(self, data_dir, **kwargs):
		super(ExtractiveDataset, self).__init__(data_dir, **kwargs)

	# Generate inputs for different models
	# @param batch: @yield of function `yield_batch`
	# @param tokenizer: Tokenizer object
	# @param model_name: See `model_name` of CLASS defined in `src.models.extractive`	
	@classmethod
	def generate_model_inputs(cls,
							  batch,
							  tokenizer,
							  model_name,
							  **kwargs,
							  ):
		if model_name == "deepset/roberta-base-squad2":
			# Unpack keyword arguments
			max_length = kwargs.get("max_length", 512)
			# Generate batch inputs
			batch_inputs = list()
			contexts = list()
			questions = list()
			for data in batch:
				context = str()
				for title, sentences in data["context"]:
					# context += title + '\n'
					context += '\n'.join(sentences) + '\n'
				contexts.append(context)
				questions.append(data["question"])
			# Note that here must be question_first, this is determined by `tokenizer.padding_side` ("right" or "left", default "right")
			# See `QuestionAnsweringPipeline.preprocess` in ./site-packages/transformers/pipelines/question_answering.py for details
			model_inputs = tokenizer(questions,
									 contexts,
									 add_special_tokens = True,
									 max_length = max_length,
									 padding = "max_length",
									 truncation = True,
									 return_overflowing_tokens = False,
									 return_tensors = "pt",
									 ) 	# Dict[input_ids: Tensor(batch_size, max_length),
										#	   attention_mask: Tensor(batch_size, max_length)]
		else:
			raise NotImplementedError(model_name)
		return model_inputs


class GenerativeDataset(BaseDataset):
	dataset_name = "Generative"
	batch_data_keys = ["context",	# List[Tuple[Str, List[Str]]], i.e. List of [title, article[sentence]]
					   "question",	# Str
					   "answers",	# List[Str]
					   ]
	def __init__(self, data_dir, **kwargs):
		super(GenerativeDataset, self).__init__(data_dir, **kwargs)

	# Generate inputs for different models
	# @param batch: @yield of function `yield_batch`
	# @param tokenizer: Tokenizer object
	# @param model_name: See `model_name` of CLASS defined in `src.models.generative`	
	@classmethod
	def generate_model_inputs(cls,
							  batch,
							  tokenizer,
							  model_name,
							  **kwargs,
							  ):
		model_inputs = None
		return model_inputs			
								  

class MultipleChoiceDataset(BaseDataset):
	dataset_name = "Multiple-choice"
	batch_data_keys = ["article",	# Str, usually
					   "question",	# Str
					   "options",	# List[Str]
					   "answer",	# Int
					   ]
	def __init__(self, data_dir, **kwargs):
		super(MultipleChoiceDataset, self).__init__(data_dir, **kwargs)

	# Generate inputs for different models
	# @param batch: @yield of function `yield_batch`
	# @param tokenizer: Tokenizer object
	# @param model_name: See `model_name` of CLASS defined in `src.models.multiple_choice`
	@classmethod
	def generate_model_inputs(cls,
							  batch,
							  tokenizer,
							  model_name,
							  **kwargs,
							  ):
		if model_name == "LIAMF-USP/roberta-large-finetuned-race":
			# Unpack keyword arguments
			max_length = kwargs.get("max_length", 512)
			# Generate batch inputs
			batch_inputs = list()
			for data in batch:
				# Unpack data
				article = data["article"]
				question = data["question"]
				option = data["options"]
				flag = question.find('_') == -1
				choice_inputs = list()
				for choice in option:
					question_choice = question + ' ' + choice if flag else question.replace('_', choice)
					inputs = tokenizer(article,
									   question_choice,
									   add_special_tokens = True,
									   max_length = max_length,
									   padding = "max_length",
									   truncation = True,
									   return_overflowing_tokens = False,
									   return_tensors = None,	# return list instead of pytorch tensor, for concatenation
									   )	# Dict[input_ids: List(max_length, ),
											#	   attention_mask: List(max_length, )]
					choice_inputs.append(inputs)
				batch_inputs.append(choice_inputs)
			# InputIds and AttentionMask
			input_ids = torch.LongTensor([[inputs["input_ids"] for inputs in choice_inputs] for choice_inputs in batch_inputs])
			attention_mask = torch.LongTensor([[inputs["attention_mask"] for inputs in choice_inputs] for choice_inputs in batch_inputs])
			model_inputs = {"input_ids": input_ids,	# (batch_size, n_option, max_length)
							"attention_mask": attention_mask,	# (batch_size, n_option, max_length)
							}
		elif model_name == "potsawee/longformer-large-4096-answering-race":
			# Unpack keyword arguments
			max_length = kwargs["max_length"]
			# Generate batch inputs
			batch_inputs = list()
			for data in batch:
				# Unpack data
				article = data["article"]
				question = data["question"]
				option = data["options"]
				article_question = [f"{question} {tokenizer.bos_token} article"] * 4
				# Tokenization
				inputs = tokenizer(article_question,
								   option,
								   max_length = max_length,
								   padding = "max_length",
								   truncation = True,
								   return_tensors = "pt",
								   ) 	# Dict[input_ids: Tensor(n_option, max_length),
										#	   attention_mask: Tensor(n_option, max_length)]
				batch_inputs.append(inputs)
			# InputIds and AttentionMask
			input_ids = torch.cat([inputs["input_ids"].unsqueeze(0) for inputs in batch_inputs], axis=0)
			attention_mask = torch.cat([inputs["attention_mask"].unsqueeze(0) for inputs in batch_inputs], axis=0)
			model_inputs = {"input_ids": input_ids,	# (batch_size, n_option, max_length)
							"attention_mask": attention_mask,	# (batch_size, n_option, max_length)
							}
		else:
			raise NotImplementedError(model_name)
		return model_inputs
