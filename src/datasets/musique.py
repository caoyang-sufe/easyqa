# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

import os
import json

from src.datasets.base import BaseDataset

class MusiqueDataset(BaseDataset):

    def __init__(self,
                 data_path,
                 ):
        super(MusiqueDataset, self).__init__()
        self.data_path = data_path

    # @param batch_size: Int
    # @param type_: Str, e.g. "run_race", "dev", "test"
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
                    batch_size,
                    type_,
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
        file_path_full = os.path.join(self.data_path, f"musique_full_v1.0_{type_}.jsonl")
        file_path_ans = os.path.join(self.data_path, f"musique_ans_v1.0_{type_}.jsonl")
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
