# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

from transformers import (pipeline,
                          AutoTokenizer,
                          AutoModel,
                          AutoModelForQuestionAnswering,
                          RobertaTokenizer,
                          RobertaForMultipleChoice,
                          LongformerTokenizer,
                          LongformerForMultipleChoice,
                          )
from src.datasets import SquadDataset
from src.models.base import BaseModel

class ChatGLM6B(BaseModel):
    # https://huggingface.co/THUDM/chatglm-6b
    # https://huggingface.co/THUDM/chatglm-6b-int4
    # https://huggingface.co/THUDM/chatglm-6b-int4-qe
    # https://huggingface.co/THUDM/chatglm-6b-int8
    # Note: The series of chatglm-6b-xxx models cannot run on CPU
    # You can quantize with `model = model.quantize(4)` or `model = model.quantize(8)` for low GPU memory
    def __init__(self, model_path, device="cuda"):
        super(ChatGLM6BTest, self).__init__(model_path, device)

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

    def load_model(self):
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True).half().to(self.device)

    # @param data		: Dict[content(Str)]
    # @return response	: Robot response
    # @return history	: Chat history
    def run(self, data, history=list()):
        return run_chatglm_6b(data, self.tokenizer, self.model, history)
