# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

from src.models.base import ExtractiveModel, GenerativeModel, MultipleChoiceModel
from src.models.extractive import RobertaBaseSquad2, RobertaBasePFHotpotqa
from src.models.generative import Chatglm, Chatglm6bInt4
from src.models.multiple_choice import RobertaLargeFinetunedRace, LongformerLarge4096AnsweringRace
