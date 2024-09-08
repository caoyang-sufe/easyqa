# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

import os
import json

from src.datasets.base import ExtractiveDataset


class SquadDataset(ExtractiveDataset):
	dataset_name = "SQuAD"
	checked_data_dirs = ["./squad1.1/train-v1.1.json",
						 "./squad1.1/dev-v1.1.json",
						 "./squad2.0/train-v2.0.json",
						 ]
	def __init__(self,
				 data_dir,
				 ):
		super(SquadDataset, self).__init__(data_dir)

	# @param batch_size: Int
	# @param type_: Str, e.g. "train", "dev"
	# @param version: Str, e.g. "1.1", "2.0"
	# @yield batch: List[Dict]
	# - context: ["University_of_Notre_Dame", ["Architecturally, the school has a Catholic character. Atop ..."]]
	# - question: "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?"
	# - answers: ["Saint Bernadette Soubirous"]
	# - answer_starts: [515]
	# - answer_ends: [541]
	# - question_id: "5733be284776f41900661182"
	def yield_batch(self,
					batch_size,
					type_,
					version,
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
				article = paragraph["context"]	# Str
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
					batch.append({"context": [[title, [article]]],	# A question has only one article, and an article has only one paragraph
								  "question": question,
								  "answers": answers,
								  "answer_starts": answer_starts,
								  "answer_ends": answer_ends,
								  "question_id": question_id,
								  })
					current_batch_size += 1
					if current_batch_size == batch_size:
						self.check_batch_data_keys(batch)
						yield batch
						batch, current_batch_size, = list(), 0
		if current_batch_size > 0:
			self.check_batch_data_keys(batch)
			yield batch
