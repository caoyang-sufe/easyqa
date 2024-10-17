# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn
# Testscripts for src.module.attention_module

import torch
from torch.nn import functional as F

from src.modules.attention import MultiHeadAttention


def testscript():
	params = {"d_model"		: 768,
			  "num_heads"	: 8,
			  "dropout_rate": 0,
			  }
	mha = MultiHeadAttention(**params)
	# Generate a small input for quick test
	N = 4
	T_q = 32
	T_k = 48
	d_model = params["d_model"]
	test_inputs = {"queries": torch.randn(N, T_q, d_model),
				   "keys"	: torch.randn(N, T_k, d_model),
				   "values"	: torch.randn(N, T_k, d_model),
				   }
	mha_output = mha(**test_inputs)
	print(mha_output.size())	# torch.Size([4, 32, 768])

	from torchsummary import summary
	summary(mha.to("cpu"), input_size=[(T_q, d_model), (T_k, d_model), (T_k, d_model)])