# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn
# Implementation of attention mechanism

import torch
from torch.nn import Module, Linear, Dropout, functional as F

# @param Q				: (N, T_q, d_k)
# @param K				: (N, T_k, d_k)
# @param V				: (N, T_k, d_v)
# @param dropout_rate	: @param p of torch.nn.functional.dropout
# @param training		: @param training of torch.nn.functional.dropout
# @return				: (N, T_q, d_v)
# PyTorch version of scaled_dot_product_attention with no mask
# Resource: https://github.com/Kyubyong/transformer

class ScaledDotProductAttention(Module):
	"""PyTorch version of scaled_dot_product_attention with no mask
	Resource: https://github.com/Kyubyong/transformer/blob/master/modules.py"""
	def __init__(self, dropout_rate=0):
		super(ScaledDotProductAttention, self).__init__()
		self.dropout_module = Dropout(p=dropout_rate)
		
	# @param Q				: (N, T_q, d_k)
	# @param K				: (N, T_k, d_k)
	# @param V				: (N, T_k, d_v)
	# @return				: (N, T_q, d_v)
	def forward(self, Q, K, V):
		return torch.bmm(self.dropout_module(F.softmax(torch.bmm(Q, K.permute(0, 2, 1)) / Q.size(-1) ** 0.5, dim=-1)), V)


class MultiHeadAttention(Module):
	"""PyTorch version of scaled_dot_product_attention with no mask
	Resource: https://github.com/Kyubyong/transformer/blob/master/modules.py"""
	def __init__(self, d_model, num_heads=8, dropout_rate=0):
		super(MultiHeadAttention, self).__init__()
		assert not d_model % num_heads, f'Param `num_heads` should be divided by `d_model`, but got num_heads={num_heads} and d_model={d_model}'
		self.size_of_head = int(d_model / num_heads)
		self.W_Q = Linear(d_model, d_model, bias=True)
		self.W_K = Linear(d_model, d_model, bias=True)
		self.W_V = Linear(d_model, d_model, bias=True)
		self.W_O = Linear(d_model, d_model, bias=True)
		self.scaled_dot_product_attention = ScaledDotProductAttention(dropout_rate=dropout_rate)
	
	# @param queries: (N, T_q, d_model)
	# @param keys	: (N, T_k, d_model)
	# @param values	: (N, T_k, d_model)
	# @return		: (N, T_q, d_model)
	def forward(self, queries, keys, values):
		print(queries.size())
		Q = self.W_Q(queries)												# Q			: (N, T_q, d_model)
		K = self.W_K(keys)													# K			: (N, T_k, d_model)
		V = self.W_V(values)												# V			: (N, T_k, d_model)
		Q_ = torch.cat(torch.split(Q, self.size_of_head, dim=-1), axis=0)	# Q_		: (h*N, T_q, d_model/h)
		K_ = torch.cat(torch.split(K, self.size_of_head, dim=-1), axis=0)	# K_		: (h*N, T_k, d_model/h)
		V_ = torch.cat(torch.split(V, self.size_of_head, dim=-1), axis=0)	# V_		: (h*N, T_k, d_model/h)
		outputs = self.scaled_dot_product_attention(Q_, K_, V_)				# outputs	: (h*N, T_q, d_model/h)
		outputs = torch.cat(torch.split(outputs, Q.size(0), dim=0), axis=2)	# outputs	: (N, T_q, d_model)
		outputs += queries													# Residual connections
		return outputs
		