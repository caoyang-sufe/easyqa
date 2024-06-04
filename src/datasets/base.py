# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

from src.base import BaseClass

class BaseDataset(BaseClass):

    def __init__(self, **kwargs):
        super(BaseDataset, self).__init__(**kwargs)

    def yield_batch(self, **kwargs):
        raise NotImplementedError()

    @classmethod
    def generate_model_inputs(cls, batch, tokenizer, **kwargs):
        raise NotImplementedError()