# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

from src.models.base import ExtractiveModel, GenerativeModel, MultipleChoiceModel
from src.models.extractive import RobertaBaseSquad2
from src.models.generative import ChatGLM
from src.models.multiple_choice import RobertaLargeFinetunedRace, LongformerLarge4096AnsweringRace
