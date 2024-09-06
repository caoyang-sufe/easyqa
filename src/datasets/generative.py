# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

import os
import re
import json
import logging

from src.datasets.base import GenerativeDataset

class HotpotqaDataset(GenerativeDataset):
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
	# - context: List[Str, List[Str]], e.g. [["The Other Side of Love", ["\"The Other Side of Love\" is a song ..."]], [..., ...], ...]
	# - question: Str, e.g. "What position on the Billboard Top 100 did Alison Moyet's late summer hit achieve?"
	# - answer: Str, e.g. "Yes" (Nonexisted in test data)
	# - question_id: Str, e.g. "5adf9ba1554299025d62a2db"
	# - supporting_facts: List[Tuple[Title(Str), SentNo(Int)]], e.g.  [['2014 S/S', 0], ['Winner (band)', 0]] (Nonexisted in test data)
	# - type: Str, e.g. "comparison" (Nonexisted in test data)
	# - level: Str, e.g. "hard" (Nonexisted in test data)
	def yield_batch(self,
					batch_size,
					filename,
					):
		batch, current_batch_size, = list(), 0
		with open(os.path.join(self.data_dir, filename), 'r', encoding="utf8") as f:
			data = json.load(f)
		for datum in data:
			id_ = datum["_id"]
			question = datum["question"]	# Str
			context = datum["context"]	# List[Tuple[Str, List[Str]]]: In HotpotQA, The first Str in Tuple is title, and the List[Str] is tokenized paragraphs
			answer = datum.get("answer")  # Nonexisted in test data
			supporting_facts = datum.get("supporting_facts")	# Nonexisted in test data
			type_ = datum.get("type")  # Nonexisted in test data
			level = datum.get("level")  # Nonexisted in test data
			batch.append({"context": context,
						  "question": question,
						  "answers": answer,
						  "question_id": id_,
						  "type": type_,
						  "level": level,
						  "supporting_facts": supporting_facts,
						  })
			current_batch_size += 1
			if current_batch_size == batch_size:
				self.check_batch_data_keys(batch)
				yield batch
				batch, current_batch_size, = list(), 0
		if current_batch_size > 0:
			self.check_batch_data_keys(batch)
			yield batch

class MusiqueDataset(GenerativeDataset):
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
	# @param category: Str, e.g. "ans", "full"
	# @param answerable: Boolean, e.g. take effect when @param category is "full"
	# @yield batch: List[Dict]
	# - id: Str, e.g. "2hop__55254_176500"
	# - context: List[Str, List[Str]]
	# - question: Str
	# - question_decomposition: List[Dict{id: Int, question: Str, answer: Str, paragraph_support_idx: Int[None]}] (Nonexisted in test data)
	# - answer: List[Str], ground truth with length only one! (Nonexisted in test data)
	# - answer_aliases: List[Str], other possible answers (Nonexisted in test data)
	# - answerable: Boolean (Nonexisted in test data)
	def yield_batch(self,
					batch_size,
					type_,
					category = "ans",
					answerable = None,
					):
		# Tool function for loading .jsonl file
		def _easy_load_jsonl(_file_path):
			_jsonl = list()
			with open(_file_path, 'r', encoding="utf8") as f:
				while True:
					_jsonl_string = f.readline()
					if not _jsonl_string:
						break
					_jsonl.append(json.loads(_jsonl_string))
			return _jsonl
		# Load data
		jsonl = _easy_load_jsonl(os.path.join(self.data_dir, f"musique_{category}_v1.0_{type_}.jsonl"))
		if category == "ans" or type_ == "test":
			sorted_jsonl = sorted(jsonl, key = lambda _json: _json["id"])
		elif category == "full":
			sorted_jsonl = sorted(jsonl, key = lambda _json: (_json["answerable"], _json["id"]))
			data_size = len(sorted_jsonl)
			if answerable is True:
				sorted_jsonl = sorted_jsonl[data_size // 2: ]
			elif answerable is False:
				sorted_jsonl = sorted_jsonl[: data_size // 2]
			elif answerable is None:
				sorted_jsonl = sorted_jsonl[:]
			else:
				assert False, f"Unexpected keyword argument `answerable`: {answerable}"
		else:
			assert False, f"Unexpected keyword argument `category`: {answerable}"
		del jsonl
		batch, current_batch_size = list(), 0
		for datum in sorted_jsonl:
			id_ = datum["id"]
			question = datum["question"]
			answer = datum.get("answer")	# Nonexisted in test data
			answer_aliases = datum.get("answer_aliases")	# Nonexisted in test data
			answerable = datum.get("answerable")	# Nonexisted in test data
			question_decomposition = datum.get("question_decomposition")	# Nonexisted in test data
			# Process paragraphs
			paragraphs = datum["paragraphs"]
			context = [[paragraph["title"], [paragraph["paragraph_text"]]] for paragraph in paragraphs]
			is_supporting = None if type_ == "test" else [paragraph["is_supporting"] for paragraph in paragraphs] 
			batch.append({"context": context,
						  "question": question,
						  "answers": answer if answer is None else [answer],
						  "question_id": id_,
						  "answer_aliases": answer_aliases,
						  "answerable": answerable,
						  "is_supporting": is_supporting,
						  "question_decomposition": question_decomposition,
						  })
			current_batch_size += 1
			if current_batch_size == batch_size:
				self.check_batch_data_keys(batch)
				yield batch
				batch, current_batch_size = list(), 0
		if current_batch_size > 0:
			self.check_batch_data_keys(batch)
			yield batch
 
class TriviaqaDataset(GenerativeDataset):
	dataset_name = "TriviaQA"
	checked_data_dirs = ["./qa/web-train.json",
						 "./qa/web-dev.json",
						 "./qa/web-test-without-answers.json",
						 "./qa/verified-web-dev.json",
						 "./qa/wikipedia-train.json",
						 "./qa/wikipedia-dev.json",
						 "./qa/wikipedia-test-without-answers.json",
						 "./qa/verified-wikipedia-dev.json",
						 "./evidence/web",
						 "./evidence/wikipedia",
						 "./triviaqa-unfiltered/unfiltered-web-train.json",
						 "./triviaqa-unfiltered/unfiltered-web-dev.json",
						 "./triviaqa-unfiltered/unfiltered-web-test-without-answers.json",
						 ]
	def __init__(self,
				 data_dir,
				 ):
		super(TriviaqaDataset, self).__init__(data_dir)

	# @param batch_size: Int
	# @param type_: Str, e.g. "train", "dev", "test", "verified"
	# @param category: Str, e.g. "web", "wikipedia"
	# @param unfiltered: Boolean, only take effect when @param category is "web", then data under `./triviaqa-unfiltered` directories will be read
	# @yield batch: List[Dict]
	def yield_batch(self,
					batch_size,
					type_,
					category,
					unfiltered = False,
					):
		# Load data
		if unfiltered:
			if type_ in ["train", "dev"]:
				data_path = os.path.join(self.data_dir, f"./triviaqa-unfiltered/unfiltered-{category}-{type_}.json")
			elif type_ == "test":
				data_path = os.path.join(self.data_dir, f"./triviaqa-unfiltered/unfiltered-{category}-test-without-answers.json")
			else:
				assert False, f"Unexpected keyword argument `type_`: {type_} for unfiltered TQA!"
		else:
			if type_ == "verified":
				data_path = os.path.join(self.data_dir, f"./qa/verified-{category}-dev.json")
			elif type_ in ["train", "dev"]:
				data_path = os.path.join(self.data_dir, f"./qa/{category}-{type_}.json")
			elif type_ == "test":
				data_path = os.path.join(self.data_dir, f"./qa/{category}-test-without-answers.json")
			else:
				assert False, f"Unexpected keyword argument `type_`: {type_}"
		with open(data_path, 'r', encoding="utf8") as f:
			data = json.load(f)["Data"]
		batch, current_batch_size, = list(), 0
		for entry in data:
			normalized_entry = self._normalize_entry(entry)
			# Generate context by EntityPages
			context = list()
			entity_title = normalized_entry["entity_title"]
			entity_filename = normalized_entry["entity_filename"]
			for title, filename in zip(entity_title, entity_filename):
				file_path = os.path.join(self.data_dir, f"./evidence/{category}", filename)
				with open(file_path, 'r', encoding="utf8") as f:
					article = list(filter(None, f.read().splitlines()))
				context.append([title, article])
			answers = normalized_entry["answer_normalized_aliases"][:]	# Simply use `answer_normalized_aliases`
			batch.append({"context": context,
						  "question": normalized_entry["question"],
						  "answers": answers,
						  })
			current_batch_size += 1
			if current_batch_size == batch_size:
				# self.check_batch_data_keys(batch)
				yield batch
				batch, current_batch_size, = list(), 0
		if current_batch_size > 0:
			# self.check_batch_data_keys(batch)
			yield batch

	# Normalize a single entry of TriviaQA data
	# @param entry: Dict, A single QA-sample in JSON format of TriviaQA
	# @return normalized_entry: Dict, Normalized QA-sample in JSON format
	def _normalize_entry(self, entry):
		normalized_columns = ["question", "question_id", "question_source"]
		# Extract raw data
		entity_pages = entry["EntityPages"]
		question = entry["Question"]
		question_id = entry["QuestionId"]
		question_source = entry["QuestionSource"]
		answer = entry.get("Answer")
		search_results = entry.get("SearchResults")
		question_part_of_verified_eval = entry.get("QuestionPartOfVerifiedEval")
		question_verified_eval_attempt = entry.get("QuestionVerifiedEvalAttempt")
		# Normalize the different dictationary
		answer_dict = self._normalize_dict_data(data=answer, prefix="answer")	# Normalize Answer
		entity_pages_dict = self._normalize_list_of_dicts_data(data=entity_pages, prefix="entity_pages")	# Normalize EntityPages
		search_results_dict = self._normalize_list_of_dicts_data(data=search_results, prefix="search_results")	# Normalize SearchResults
		# Combine the normalized data
		# normalized_entry = {column: eval(column) for column in normalized_columns}	# Error! Local variables (question, question_id, question_source) are not defined
		normalized_entry = dict()
		for column in normalized_columns:
			normalized_entry[column] = eval(column)
		normalized_entry = {**normalized_entry, **answer_dict, **entity_pages_dict, **search_results_dict}
		return normalized_entry
		
	# Normalize Dict-like data, e.g. Answer
	# @param data: Dict-like variable
	# @param prefix: Normalize key name by adding prefix, i.e. "answer" for Answer
	# @return normalized_dict: Dict[Obj]
	def _normalize_dict_data(self, data, prefix):
		normalized_dict = dict()
		if data is not None:
			for key, value in data.items():
				print(key, value)
				normalized_key = f"{prefix}_{self._transform_camel_to_underscore(key)}"
				normalized_dict[normalized_key] = value
		return normalized_dict

	# Normalize List[Dict]-like data, e.g. EntityPages and SearchResults
	# @param data: List[Dict]-like variable
	# @param prefix: Normalize key name by adding prefix, i.e. "entity_pages" for EntityPages and "search_results" for SearchResults
	# @return normalized_dict: Dict[List[Obj]]
	def _normalize_list_of_dicts_data(self, data, prefix):
		normalized_dict = dict()
		if data is not None:
			for i, datum in enumerate(data):	
				for key, value in datum.items():
					normalized_key = f"{prefix}_{self._transform_camel_to_underscore(key)}"
					if normalized_key in normalized_dict:
						normalized_dict[normalized_key].append(value)
					else:
						# Note that if i > 0, then it means that `datum` in `data` has different keys
						normalized_dict[normalized_key] = [None] * i + [value]
						logging.warning(f"New key occurs: {normalized_key}")
		return normalized_dict

	# Transform UpperCamelCase string to lower_case_with_underscores
	# @param string: String in UpperCamelCase format, e.g. QuestionPartOfVerifiedEval
	# @return: String in lower_case_with_underscores format, e.g. question_part_of_verified_eval
	def _transform_camel_to_underscore(self, string):
		return string[0].lower() + re.sub("[A-Z]", lambda _match: f"_{_match.group().lower()}", string[1:])
