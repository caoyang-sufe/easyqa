# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn
# Commonly used module

import torch
from torch.autograd import Variable
from torch.nn import Module, LSTM, Dropout, functional as F

# @param x			: (size_1, size_2, ... , size_n)
# @param x_shape	: (size_1, )
# @return x_masked	: (size_1, size_2, ... , size_n)
# @return x_mask	: (size_1, size_2, ... , size_n) with value 0 or 1
def mask(x, x_shape):
	x_mask = x.new(x.size()).zero_()
	for i in range(x_shape.size(0)):
		x_mask[i, :x_shape[i].item()] = 1
	x_mask = Variable(x_mask, requires_grad=False)
	x_masked = x * x_mask
	return x_masked, x_mask

# @param x			: (size_1, size_2, ... , size_n)
# @param x_shape	: (size_1, )
# @return			: (size_1, size_2, ... , size_n)
def masked_softmax(x, x_shape, dim=-1, epsilon=1e-13):
	x_masked, x_mask = mask(x, x_shape)
	x_softmax_numerator = F.softmax(x_masked, dim=dim) * mask
	x_softmax_denominator = torch.sum(x_softmax_numerator, dim=dim, keepdim=True) + epsilon
	return x_softmax_numerator / x_softmax_denominator


class MaskedLSTM(Module):
	"""LSTM Module which deals with masked inputs"""
	def __init__(self,
				 input_size,
				 hidden_size,
				 num_layers,
				 batch_first,
				 bidirectional,
				 dropout):
		super(MaskedLSTM, self).__init__()
		self.lstm = LSTM(input_size		= input_size,
						 hidden_size	= hidden_size,
						 num_layers		= num_layers,
						 batch_first	= batch_first,
						 bidirectional	= bidirectional,
						 dropout		= dropout)
		self.dropout = Dropout(dropout)

	# @param x			: (batch_size, sequence_length, feature_size)
	# @param x_shape	: (batch_size, )
	def forward(self, x, x_shape):
		x_masked, x_mask = mask(x, x_shape)
		x_drop = self.dropout(x_masked)
		x_hidden, _ = self.lstm(x_drop)
		x_hidden_masked, x_hidden_mask = mask(x_hidden, x_shape)
		return x_hidden_masked