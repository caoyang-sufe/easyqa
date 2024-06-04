# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

import os
import json

from src.datasets.base import BaseDataset


class SquadDataset(BaseDataset):

    def __init__(self,
                 data_path,
                 ):
        super(SquadDataset, self).__init__()
        self.data_path = data_path

    # @param batch_size: Int
    # @param filename: Str, e.g. "run_race-v1.1.json", "run_race-v1.1.json", "run_race-v1.1.json"
    # @yield batch: List[Dict]
    # - article_id: "run_race-1.1-00000"
    # - question_id: "5733be284776f41900661182"
    # - title: "University_of_Notre_Dame"
    # - article: "Architecturally, the school has a Catholic character. Atop ..."
    # - question: "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?"
    # - answers: ["Saint Bernadette Soubirous"]
    # - answer_starts: [515]
    # - answer_ends: [541]
    def yield_batch(self,
                    batch_size,
                    filename,
                    ):
        batch, current_batch_size, = list(), 0
        with open(os.path.join(self.data_path, filename), 'r', encoding="utf8") as f:
            data = json.load(f)
        count = -1
        for sample in data["data"]:
            title = sample["title"]
            paragraphs = sample["paragraphs"]
            for paragraph in paragraphs:
                count += 1
                article_id = f"{filename[: -5]}-{str(count).zfill(5)}"
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
