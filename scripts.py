# -*- coding: utf-8 -*-
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn
# Useful scripts

import os
import gc
import torch

from settings import DATA_DIR, LOG_DIR, MODEL_ROOT, DATA_SUMMARY, MODEL_SUMMARY

from src.datasets import RaceDataset, DreamDataset, SquadDataset, HotpotqaDataset, MusiqueDataset, TriviaqaDataset
from src.models import RobertaLargeFinetunedRace, LongformerLarge4096AnsweringRace, RobertaBaseSquad2, Chatglm6bInt4
from src.pipelines import RacePipeline, DreamPipeline, SquadPipeline
from src.tools.easy import initialize_logger, terminate_logger

def script_1():
	pipeline = SquadPipeline()
	pipeline.easy_inference_pipeline(model_class_name)
	