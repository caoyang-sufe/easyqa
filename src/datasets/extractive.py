# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

import os
import json

from src.datasets.base import BaseExtractiveDataset


class SquadDataset(BaseExtractiveDataset):
	dataset_name = "SQuAD"
	checked_data_dirs = ["./squad1.1/train-v1.1",
						 "./squad1.1/dev-v1.1.json",
						 "./squad2.0/train-v2.0",
						 ]
	def __init__(self,
				 data_dir,
				 ):
		super(SquadDataset, self).__init__(data_dir)

	# @param batch_size: Int
	# @param version: Str, e.g. "1.1", "2.0"
	# @param type_: Str, e.g. "train", "dev"
	# @yield batch: List[Dict]
	# - article_id: "train-1.1-00000"
	# - question_id: "5733be284776f41900661182"
	# - title: "University_of_Notre_Dame"
	# - article: "Architecturally, the school has a Catholic character. Atop ..."
	# - question: "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?"
	# - answers: ["Saint Bernadette Soubirous"]
	# - answer_starts: [515]
	# - answer_ends: [541]
	def yield_batch(self,
					batch_size,
					version,
					type_,
					):
		batch, current_batch_size, = list(), 0
		with open(os.path.join(self.data_dir, f"squad{version}", f"{type_}-v{version}.json"), 'r', encoding="utf8") as f:
			data = json.load(f)
		count = -1
		for sample in data["data"]:
			title = sample["title"]
			paragraphs = sample["paragraphs"]
			for paragraph in paragraphs:
				count += 1
				article_id = f"{type_}-v{version}-{str(count).zfill(5)}"
				article = paragraph["context"]
				for qas in paragraph["qas"]:
					question_id = qas["id"]
					question = qas["question"]
					candidate_answers = qas["answers"]
					answer_starts, answer_ends, answers = list(), list(), list()
					for candidate_answer in candidate_answers:
						answer_start = int(candidate_answer["answer_start"])
						answer = candidate_answer["text"]
						answer_end = answer_start + len(answer)
						assert answer == article[answer_start: answer_end]
						answer_starts.append(answer_start)
						answer_ends.append(answer_end)
						answers.append(answer)
					batch.append({"article_id": article_id,
								  "question_id": question_id,
								  "title": title,
								  "article": article,
								  "question": question,
								  "answers": answers,
								  "answer_starts": answer_starts,
								  "answer_ends": answer_ends,
								  })
					current_batch_size += 1
					if current_batch_size == batch_size:
						yield batch
						batch, current_batch_size, = list(), 0
		if current_batch_size > 0:
			yield batch



class HotpotqaDataset(BaseExtractiveDataset):
	dataset_name = "HotpotQA"
	checked_data_dirs = ["./hotpot_dev_distractor_v1.json",
						 "./hotpot_dev_fullwiki_v1.json",
						 "./hotpot_test_fullwiki_v1.json",
						 "./hotpot_train_v1.1.json",
						 ]
	def __init__(self,
				 data_dir,
				 ):
		super(HotpotqaDataset, self).__init__(data_dir)

	# @param batch_size: Int
	# @param filename: Str, e.g. "train_v1.1.json", "dev_distractor_v1.json", "dev_fullwiki_v1.json", "test_fullwiki_v1.json"
	# @yield batch: List[Dict]
	# - id: "5adf9ba1554299025d62a2db"
	# - question: "What position on the Billboard Top 100 did Alison Moyet's late summer hit achieve?"
	# - context: List[Tuple[Str, List[Str]]]: In HotpotQA, The first Str in Tuple is title, and the List is tokenized sentences
	#   e.g. [["The Other Side of Love", ["\"The Other Side of Love\" is a song ...", ]], [..., [...]], ...]
	# - answer: "Yes" (may be None in test file)
	# - type: "comparison"
	# - level: "hard"
	def yield_batch(self,
					batch_size,
					filename,
					):
		batch, current_batch_size, = list(), 0
		with open(os.path.join(self.data_dir, filename), 'r', encoding="utf8") as f:
			data = json.load(f)
		for sample in data:
			id_ = sample["_id"]
			question = sample["question"]
			context = sample["context"]
			answer = sample.get("answer")  # Maynot in test file
			type_ = sample.get("type")  # Maynot in test file
			level = sample.get("level")  # Maynot in test file
			batch.append({"id": id_,
						  "question": question,
						  "context": context,
						  "answer": answer,
						  "type": type_,
						  "level": level,
						  })
			current_batch_size += 1
			if current_batch_size == batch_size:
				yield batch
				batch, current_batch_size, = list(), 0
		if current_batch_size > 0:
			yield batch

	# Generate inputs for given model
	@classmethod
	def generate_model_inputs(cls,
							  batch,
							  tokenizer,
							  model_name="AdapterHub/roberta-base-pf-hotpotqa",
							  **kwargs,
							  ):
		if model_name == "AdapterHub/roberta-base-pf-hotpotqa":
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



class MusiqueDataset(BaseExtractiveDataset):
	dataset_name = "Musique"
	checked_data_dirs = ["./musique_ans_v1.0_train.jsonl",
						 "./musique_ans_v1.0_dev.jsonl",
						 "./musique_ans_v1.0_test.jsonl",
						 "./musique_full_v1.0_train.jsonl",
						 "./musique_full_v1.0_dev.jsonl",
						 "./musique_full_v1.0_test.jsonl",
						 ]
	def __init__(self,
				 data_dir,
				 ):
		super(MusiqueDataset, self).__init__(data_dir)

	# @param batch_size: Int
	# @param type_: Str, e.g. "train", "dev", "test"
	# @yield batch: List of `{"full": [json_full_1, json_full_2], "ans": json_ans}`, where `json_full_1, json_full_2, json_ans` have the same key-value pairs
	# JSON structure in @yield batch is as below:
	# - id: Str, e.g. "2hop__55254_176500"
	# - paragraphs: List[Dict{idx: Int, title: Str, paragraph_text: Str, is_supporting: Boolean}]
	# - question: Str
	# - question_decomposition: List[Dict{id: Int, question: Str, answer: Str, paragraph_support_idx: Int[None]}]
	# - answer: Str, ground truth
	# - answer_aliases: List[Str], other possible answers
	# - answerable: Boolean
	def yield_batch(self,
					batch_size: int,
					type_: str,
					):
		"""Tool function for loading .jsonl file"""
		def _easy_load_jsonl(_file_path):
			_jsonl = list()
			with open(_file_path, 'r', encoding="utf8") as f:
				while True:
					_jsonl_string = f.readline()
					if not _jsonl_string:
						break
					_jsonl.append(json.loads(_jsonl_string))
			return _jsonl
		file_path_full = os.path.join(self.data_dir, f"musique_full_v1.0_{type_}.jsonl")
		file_path_ans = os.path.join(self.data_dir, f"musique_ans_v1.0_{type_}.jsonl")
		jsonl_full = _easy_load_jsonl(file_path_full)
		jsonl_ans = _easy_load_jsonl(file_path_ans)
		sorted_jsonl_full = sorted(jsonl_full, key = lambda _json: _json["id"])
		sorted_jsonl_ans = sorted(jsonl_ans, key = lambda _json: _json["id"])
		del jsonl_full, jsonl_ans

		batch, current_batch_size = list(), 0
		for i in range(len(sorted_jsonl_ans)):
			# Sampling and sanity check
			json_full_1 = sorted_jsonl_full[2 * i]
			json_full_2 = sorted_jsonl_full[2 * i + 1]
			json_ans = sorted_jsonl_ans[i]
			id_full_1 = json_full_1["id"]
			id_full_2 = json_full_2["id"]
			id_ans = json_ans["id"]
			assert id_full_1 == id_full_2 == id_ans, f"Mismatch: {id_full_1} v.s. {id_full_2} v.s. {id_ans}"
			batch.append({"full": [json_full_1, json_full_2], "ans": json_ans})
			current_batch_size += 1
			if current_batch_size == batch_size:
				yield batch
				batch, current_batch_size = list(), 0
		if current_batch_size > 0:
			yield batch

	def generate_model_inputs(cls, batch, tokenizer, **kwargs):
		pass
