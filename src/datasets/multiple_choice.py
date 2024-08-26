# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

import os
import json
import torch

from src.datasets.base import BaseMultipleChoiceDataset


class RaceDataset(BaseMultipleChoiceDataset):
	checked_data_dirs = ["./train/high/",
						 "./train/middle/",
						 "./dev/high/",
						 "./dev/middle/",
						 "./test/high/",
						 "./test/middle/",
						 ]
	def __init__(self,
				 data_dir,
				 ):
		super(RaceDataset, self).__init__(data_dir)
		self.data_dir = data_dir

	# @param batch_size: Int
	# @param types: List[Str] of "train", "dev", "test"
	# @param difficulties: List[Str] of "high", "middle"
	# @yield batch: List[Dict]
	# @return "article_id": "high1.txt"
	# @return "question_id": 0
	# @return "article": "My husband is a born shopper. He ..."
	# @return "question": "The husband likes shopping because   _  ."
	# @return "options": ["he has much money.", "he likes the shops.", "he likes to compare the prices between the same items.", "he has nothing to do but shopping."]
	# @return "answer": 2
	def yield_batch(self,
					batch_size,
					types,
					difficulties,
					):
		batch, current_batch_size = list(), 0
		for type_ in types:
			for difficulty in difficulties:
				data_dir = os.path.join(self.data_dir, type_, difficulty)
				for filename in os.listdir(data_dir):
					with open(os.path.join(data_dir, filename), 'r', encoding="utf8") as f:
						data = json.load(f)
					article_id = data["id"]
					article = data["article"]
					questions = data["questions"]
					options = data["options"]
					answers = data["answers"]
					# Ensure the number of questions, options and answers are the same
					n_questions = len(questions)
					n_options = len(options)
					n_answers = len(answers)
					assert n_options == n_questions == n_answers, f"Inconsistent of the number of questions({n_questions}), options({n_options}) and answers({n_answers})!"
					# Ensure id in data matches filename
					assert article_id == difficulty + filename, f"Inconsistent of id and filename: {article_id} v.s. {difficulty + filename}!"
					for question_id, (question, option, answer) in enumerate(zip(questions, options, answers)):
						# Ensure the number of option is 4
						n_option = len(option)
						assert n_option == 4, f"There are {n_option} options in the {question_id}th question of article {article_id}!"
						batch.append({"article_id": article_id,
									  "question_id": question_id,
									  "article": article,
									  "question": question,
									  "options": option,
									  "answer": "ABCD".index(answer),
									  })
						current_batch_size += 1
						if current_batch_size == batch_size:
							yield batch
							batch, current_batch_size = list(), 0
		if current_batch_size > 0:
			yield batch

	# Generate inputs for different models
	# @20240528: LIAMF-USP/roberta-large-finetuned-race
	# @20240528: potsawee/longformer-large-4096-answering-race
	@classmethod
	def generate_model_inputs(cls,
							  batch,
							  tokenizer,
							  model_name="LIAMF-USP/roberta-large-finetuned-race",
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


class DreamDataset(BaseMultipleChoiceDataset):
	checked_data_dirs = ["./train.json", "./dev.json", "./test.json"]
	def __init__(self,
				 data_dir,
				 ):
		super(DreamDataset, self).__init__(data_dir)
		self.data_dir = data_dir

	# @param batch_size: Int
	# @param types: List[Str] of "train", "dev", "test"
	# @yield batch: List[Dict]
	# - "article_id": "5-510"
	# - "question_id": 0
	# - "article": ""M: I am considering dropping my dancing class ..."
	# - "question": "What does the man suggest the woman do?"
	# - "options": ["Consult her dancing teacher.", "Take a more interesting class.", "Continue her dancing class."]
	# - "answer": 0
	def yield_batch(self,
					batch_size,
					types,
					):
		batch, current_batch_size,  = list(), 0
		for type_ in types:
			with open(os.path.join(self.data_dir, f"{type_}.json"), 'r', encoding="utf8") as f:
				data = json.load(f)
			for (article_sentences, questions, article_id) in data:
				article = '\n'.join(article_sentences)	# Conversation-like article is split by '\n', simply concatenate them
				for question_id, question_data in enumerate(questions):
					question = question_data["question"]
					choice = question_data["choice"]
					flag = False
					for index, option in enumerate(choice):
						if option == question_data["answer"]:
							assert not flag, f"Two same options in question {id_}"
							answer = index
							flag = True
					assert flag, f"No option matching answer in question {id_}"
					batch.append({"article_id": article_id,
								  "question_id": question_id,
								  "article": article,
								  "question": question,
								  "options": choice,
								  "answer": answer,
								  })
					current_batch_size += 1
					if current_batch_size == batch_size:
						yield batch
						batch, current_batch_size = list(), 0
		if current_batch_size > 0:
			yield batch

	# Generate inputs for different models
	# @20240528: LIAMF-USP/roberta-large-finetuned-race
	# @20240528: potsawee/longformer-large-4096-answering-race
	@classmethod
	def generate_model_inputs(cls,
							  batch,
							  tokenizer,
							  model_name="LIAMF-USP/roberta-large-finetuned-race",
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
