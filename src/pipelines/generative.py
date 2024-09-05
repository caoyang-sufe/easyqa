# -*- coding: utf-8 -*- 
# @author : caoyang
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

