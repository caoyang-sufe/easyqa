# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

import gc
import time
from transformers import (pipeline,
                          AutoTokenizer,
                          AutoModel,
                          AutoModelForQuestionAnswering,
                          RobertaTokenizer,
                          RobertaForMultipleChoice,
                          LongformerTokenizer,
                          LongformerForMultipleChoice,
                          )

from src.pipelines.base import BasePipeline
from src.datasets import RaceDataset, DreamDataset

class MultipleChoicePipeline(BasePipeline):

    def __init__(self):
        super(MultipleChoicePipeline, self).__init__()

    def run_race(self,
                 Model,
                 data_path,
                 model_path,
                 device,
                 max_length,
                 ):
        dataset = RaceDataset(data_path = data_path)
        model = Model(model_path = model_path, device = device)
        for batch in dataset.yield_batch(batch_size = 2,
                                         types = ["train"],
                                         difficulties = ["high"],
                                         ):
            batch_logits, batch_predicts = model.run(batch = batch, max_length = max_length)
            print(batch_logits, batch_predicts)
            input("Pause ...")

    def run_dream(self,
                  Model,
                  data_path,
                  model_path,
                  device,
                  max_length,
                  ):
        dataset = DreamDataset(data_path = data_path)
        model = Model(model_path = model_path, device = device)
        for batch in dataset.yield_batch(batch_size = 2,
                                         types = ["train"],
                                         ):
            batch_logits, batch_predicts = model.run(batch = batch, max_length = max_length)
            print(batch_logits, batch_predicts)
