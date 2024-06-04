# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

from src.base import BaseClass
from transformers import AutoTokenizer, AutoModel

class BaseModel(BaseClass):
    Tokenizer = AutoTokenizer
    Model = AutoModel

    def __init__(self, **kwargs):
        super(BaseModel, self).__init__(**kwargs)
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()

    def load_tokenizer(self):
        tokenizer = self.Tokenizer.from_pretrained(self.model_path)
        return tokenizer

    def load_model(self):
        model = self.Model.from_pretrained(self.model_path).to(self.device)
        return model
