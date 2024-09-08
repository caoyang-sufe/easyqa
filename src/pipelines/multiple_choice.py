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

from src.pipelines.base import MultipleChoicePipeline
from src.datasets import RaceDataset, DreamDataset
from src.models import RobertaLargeFinetunedRace, LongformerLarge4096AnsweringRace
from settings import DATA_SUMMARY, MODEL_SUMMARY


class RacePipeline(MultipleChoicePipeline):

	def __init__(self):
		super(RacePipeline, self).__init__()



class DreamPipeline(MultipleChoicePipeline):

    def __init__(self):
        super(MultipleChoicePipeline, self).__init__()