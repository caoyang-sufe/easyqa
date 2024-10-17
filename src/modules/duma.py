# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn
# Implementation of DUMA

import torch
from torch.nn import Module, Linear, NLLLoss, functional as F

from src.modules.easy import load_model
from src.modules.attention import MultiHeadAttention

from settings import MODEL_SUMMARY

class DUMA(Module):
	"""Forward propagation algorithm of DCMN+
	Reference: [DUMA: Reading Comprehension with Transposition Thinking](https://arxiv.org/abs/2001.09415v5)
	Notice here we do not do sentence tokenization in DUMA (also in HRCA), but directly input the whole article tokens into model, which is different from Co-Matching and DCMN+
	This is feasible for DREAM, but in RACE, the number of total article tokens may be 1000+, which is hard for training.
	- Input:
	  - $P \in \R^{d×p}$ 
	  - $Q \in \R^{d×q}$
	  - $A = \{A_1, ..., A_m\}, A_j \in \R^{d×a}$
	- Output:
	  - $L \in \R^m$
	- Example:
	>>> args = load_args(Config=ModuleConfig)
	>>> kwargs = {"train_batch_size"		: 8,
				  "max_article_token"		: 128,
				  "max_question_token"		: 16,
				  "max_option_token"		: 24,
				  "duma_fuse_method"		: None,	# Change as "mul", "sum", "cat"
				  "duma_num_layers"			: 2,
				  "duma_mha_num_heads"		: 8,
				  "duma_mha_dropout_rate"	: 0.,
				  "duma_pretrained_model"	: "albert-base-v1",
				  "duma_encoding_size"		: 768}
	>>> update_args(args, **kwargs)
	>>> P_size = (args.train_batch_size, args.max_article_token)
	>>> Q_size = (args.train_batch_size, args.max_question_token)
	>>> A_size = (args.train_batch_size * N_CHOICES, args.max_option_token)
	>>> test_input = {'P'	: {"input_ids"		: (torch.randn(*P_size).abs() * 10).long(),
							   "token_type_ids"	: torch.zeros(*P_size).long(),
						   	   "attention_mask"	: torch.ones(*P_size).long()},
					  'Q'	: {"input_ids"		: (torch.randn(*Q_size).abs() * 10).long(),
						   	   "token_type_ids"	: torch.zeros(*Q_size).long(),
						   	   "attention_mask"	: torch.ones(*Q_size).long()},
					  'A'	: {"input_ids"		: (torch.randn(*A_size).abs() * 10).long(),
						   	   "token_type_ids"	: torch.zeros(*A_size).long(),
						   	   "attention_mask"	: torch.ones(*A_size).long()},
					  }
	>>> duma = DUMA(args=args)
	>>> duma_output = duma.forward(**test_input)"""
	loss_function = NLLLoss()
	def __init__(self, args):
		super(DUMA, self).__init__()
		self.device = args.device
		self.p = args.max_article_token
		self.q = args.max_question_token
		self.a = args.max_option_token
		self.m = args.n_choices
		self.l = args.duma_encoding_size
		self.k = args.duma_num_layers
		self.fuse_method = args.duma_fuse_method

		self.multi_head_attention = MultiHeadAttention(d_model=args.duma_encoding_size, num_heads=args.duma_mha_num_heads, dropout_rate=args.duma_mha_dropout_rate)
		self.fuse_linear_x = Linear(self.l, self.l, bias=True)
		self.fuse_linear_y = Linear(self.l, self.l, bias=True)
		if self.fuse_method in ["mul", "sum"]:
			self.W = Linear(self.l, 1, bias=False)
		elif self.fuse_method == "cat":
			self.W = Linear(2 * self.l, 1, bias=False)
		else:
			self.W = Linear(self.l, 1, bias=False)
		if args.load_pretrained_model_in_module:
			self.pretrained_model = load_model(
				model_path = MODEL_SUMMARY[args.duma_pretrained_model]["path"],
				device = args.pretrained_model_device,
			)
			self.pretrained_model.eval()
		else:
			self.pretrained_model = None
	# @param P	: {"input_ids": tensor, "token_type_ids": tensor, "attention_mask": tensor}, tensor(batch_size, max_article_token)
	# @param Q	: {"input_ids": tensor, "token_type_ids": tensor, "attention_mask": tensor}, tensor(batch_size, max_question_token)
	# @param A	: {"input_ids": tensor, "token_type_ids": tensor, "attention_mask": tensor}, tensor(batch_size * N_CHOICES, max_option_token)
	def forward(self, P, Q, A, pretrained_model=None):
		E_P, E_QA = self.encoder(P, Q, A, pretrained_model=pretrained_model)
		O = self.dual_multi_head_co_attention(E_P, E_QA)	# O: (batch_size, N_CHOICES, ?)
		L = self.decoder(O)									# L: (batch_size, N_CHOICES)
		return L

	# @param P		: {"input_ids": tensor, "token_type_ids": tensor, "attention_mask": tensor}, tensor(batch_size, max_article_token)
	# @param Q		: {"input_ids": tensor, "token_type_ids": tensor, "attention_mask": tensor}, tensor(batch_size, max_question_token)
	# @param A		: {"input_ids": tensor, "token_type_ids": tensor, "attention_mask": tensor}, tensor(batch_size * N_CHOICES, max_option_token)
	# @return E_P	: (batch_size, N_CHOICES, max_article_token, duma_encoding_size)
	# @return E_QA	: (batch_size, N_CHOICES, max_question_token + max_option_token, duma_encoding_size)
	def encoder(self, P, Q, A, pretrained_model=None):
		batch_size = P["input_ids"].size(0)
		size_of_split_choice = (batch_size, self.m, self.a)
		A["input_ids"] = A["input_ids"].view(*size_of_split_choice)
		A["token_type_ids"] = A["token_type_ids"].view(*size_of_split_choice)
		A["attention_mask"] = A["input_ids"].view(*size_of_split_choice)
		E_list = list()
		for i in range(self.m):
			concat_inputs = {"input_ids"		: torch.cat([P["input_ids"], Q["input_ids"], A["input_ids"][:, i, :]], axis=-1),				# (batch_size, max_article_token + max_question_token + max_option_token)
							 "token_type_ids"	: torch.cat([P["token_type_ids"], Q["token_type_ids"], A["token_type_ids"][:, i, :]], axis=-1),	# (batch_size, max_article_token + max_question_token + max_option_token)
							 "attention_mask"	: torch.cat([P["attention_mask"], Q["attention_mask"], A["attention_mask"][:, i, :]], axis=-1),	# (batch_size, max_article_token + max_question_token + max_option_token)
							 }
			E_list.append(pretrained_model(**concat_inputs).last_hidden_state.unsqueeze(1) if self.pretrained_model is None else self.pretrained_model(**concat_inputs).last_hidden_state.unsqueeze(1))
		E = torch.cat(E_list, axis=1)				# E		: (batch_size, N_CHOICES, max_article_token + max_question_token + max_option_token, duma_encoding_size)
		E_P = E[:, :, :self.p, :]					# E_P	: (batch_size, N_CHOICES, max_article_token, duma_encoding_size)
		E_QA = E[:, :, self.p:, :]					# E_QA	: (batch_size, N_CHOICES, max_question_token + max_option_token, duma_encoding_size)
		return E_P.to(self.device), E_QA.to(self.device)

	# @param E_P	: (batch_size, N_CHOICES, max_article_token, duma_encoding_size)
	# @param E_QA	: (batch_size, N_CHOICES, max_question_token + max_option_token, duma_encoding_size)
	# @return O		: (batch_size, N_CHOICES, ?) where ? could be duma_encoding_size or 2 * duma_encoding_size
	def dual_multi_head_co_attention(self, E_P, E_QA):
		O_list = list()
		for i in range(self.m):
			E_P_i = E_P[:, i, :, :]															# E_P_i	: (batch_size, max_article_token, duma_encoding_size)
			E_QA_i = E_QA[:, i, :, :]														# E_QA_i: (batch_size, max_question_token + max_option_token, duma_encoding_size)
			MHA_1 = self.multi_head_attention(queries=E_P_i, keys=E_QA_i, values=E_QA_i)	# MHA_1	: (batch_size, max_article_token, duma_encoding_size)
			MHA_2 = self.multi_head_attention(queries=E_QA_i, keys=E_P_i, values=E_P_i)		# MHA_2	: (batch_size, max_question_token + max_option_token, duma_encoding_size)
			if self.k > 1:
				# Stack k layers
				for _ in range(self.k - 1):
					MHA_1 = self.multi_head_attention(queries=MHA_1, keys=MHA_2, values=MHA_2)	# MHA_1	: (batch_size, max_article_token, duma_encoding_size)
					MHA_2 = self.multi_head_attention(queries=MHA_2, keys=MHA_1, values=MHA_1)	# MHA_2	: (batch_size, max_question_token + max_option_token, duma_encoding_size)
			O_i = self._fuse(x=MHA_1, y=MHA_2)													# O_i	: (batch_size, ?)
			O_list.append(O_i.unsqueeze(1))
		O = torch.cat(O_list, axis=1)															# O		: (batch_size, N_CHOICES, ?)
		return O

	# @param O		: (batch_size, N_CHOICES, ?) where ? could be duma_encoding_size or 2 * duma_encoding_size
	# @return L		: (batch_size, N_CHOICES)
	def decoder(self, O):
		L_unactived = self.W(O).squeeze(-1)						# L_unactived	: (batch_size, N_CHOICES)
		L = F.log_softmax(L_unactived, dim=-1)					# L				: (batch_size, N_CHOICES)
		return L

	# @param x	: (batch_size, x_length, duma_encoding_size)
	# @param y	: (batch_size, y_length, duma_encoding_size)
	# @return	: (batch_size, ?) where ? could be duma_encoding_size or 2 * duma_encoding_size
	# I don not known the concrete implementation of the Fuse function in origin paper, as the author did not provide formulas or codes
	def _fuse(self, x, y):
		x_project = self.fuse_linear_x(x)					# x_project			: (batch_size, x_length, duma_encoding_size)
		y_project = self.fuse_linear_x(y)					# y_project			: (batch_size, y_length, duma_encoding_size)
		x_project_pooled = torch.max(x_project, axis=1)[0]	# x_project_pooled	: (batch_size, duma_encoding_size)
		y_project_pooled = torch.max(y_project, axis=1)[0]	# y_project_pooled	: (batch_size, duma_encoding_size)
		if self.fuse_method == "mul":
			return torch.sigmoid(x_project_pooled * y_project_pooled)						# @return	: (batch_size, duma_encoding_size)
		elif self.fuse_method == "sum":
			return torch.sigmoid(x_project_pooled + y_project_pooled)						# @return	: (batch_size, duma_encoding_size)
		elif self.fuse_method == "cat":
			return torch.sigmoid(torch.cat([x_project_pooled, y_project_pooled], axis=-1))	# @return	: (batch_size, 2 * duma_encoding_size)
		else:
			# Inspired from FuseNet in https://github.com/Qzsl123/dcmn/blob/master/dcmn.py
			p = torch.sigmoid(x_project_pooled + y_project_pooled)			# p			: (batch_size, duma_encoding_size)
			return p * x_project_pooled + (1 - p) * y_project_pooled		# @return	: (batch_size, duma_encoding_size)


class DUMAv1(DUMA):
	"""Encode passage(P) and question-and-answer(QA) respectively"""
	def __init__(self, args):
		super(DUMAv1, self).__init__(args)

	# @param P		: {"input_ids": tensor, "token_type_ids": tensor, "attention_mask": tensor}, tensor(batch_size, max_article_token)
	# @param Q		: {"input_ids": tensor, "token_type_ids": tensor, "attention_mask": tensor}, tensor(batch_size, max_question_token)
	# @param A		: {"input_ids": tensor, "token_type_ids": tensor, "attention_mask": tensor}, tensor(batch_size * N_CHOICES, max_option_token)
	# @return E_P	: (batch_size, N_CHOICES, max_article_token, duma_encoding_size)
	# @return E_QA	: (batch_size, N_CHOICES, max_question_token + max_option_token, duma_encoding_size)
	def encoder(self, P, Q, A, pretrained_model=None):
		batch_size = P["input_ids"].size(0)
		size_of_split_choice = (batch_size, self.m, self.a)
		A["input_ids"] = A["input_ids"].view(*size_of_split_choice)
		A["token_type_ids"] = A["token_type_ids"].view(*size_of_split_choice)
		A["attention_mask"] = A["input_ids"].view(*size_of_split_choice)
		E_QA_list = list()
		for i in range(self.m):
			concat_inputs = {"input_ids"		: torch.cat([Q["input_ids"], A["input_ids"][:, i, :]], axis=-1),			# (batch_size, max_question_token + max_option_token)
							 "token_type_ids"	: torch.cat([Q["token_type_ids"], A["token_type_ids"][:, i, :]], axis=-1),	# (batch_size, max_question_token + max_option_token)
							 "attention_mask"	: torch.cat([Q["attention_mask"], A["attention_mask"][:, i, :]], axis=-1),	# (batch_size, max_question_token + max_option_token)
							 }
			E_QA_list.append(pretrained_model(**concat_inputs).last_hidden_state.unsqueeze(1) if self.pretrained_model is None else self.pretrained_model(**concat_inputs).last_hidden_state.unsqueeze(1))
		E_QA = torch.cat(E_QA_list, axis=1)																										# E_QA			: (batch_size, N_CHOICES, max_question_token + max_option_token, duma_encoding_size)
		E_P_unrepeated = pretrained_model(**P).last_hidden_state if self.pretrained_model is None else pretrained_model(**P).last_hidden_state	# E_P_unrepeated: (batch_size, max_article_token, duma_encoding_size)
		E_P = E_P_unrepeated.unsqueeze(1).repeat(1, self.m, 1, 1)																				# E_P			: (batch_size, N_CHOICES, max_article_token, duma_encoding_size)
		return E_P.to(self.device), E_QA.to(self.device)


class DUMAv2(DUMAv1):
	"""Adding residual connection to dual multi-head co-attention"""
	def __init__(self, args):
		super(DUMAv2, self).__init__(args)

	# @param E_P	: (batch_size, N_CHOICES, max_article_token, duma_encoding_size)
	# @param E_QA	: (batch_size, N_CHOICES, max_question_token + max_option_token, duma_encoding_size)
	# @return O		: (batch_size, N_CHOICES, ?) where ? could be duma_encoding_size or 2 * duma_encoding_size
	def dual_multi_head_co_attention(self, E_P, E_QA):
		O_list = list()
		for i in range(self.m):
			E_P_i = E_P[:, i, :, :]															# E_P_i	: (batch_size, max_article_token, duma_encoding_size)
			E_QA_i = E_QA[:, i, :, :]														# E_QA_i: (batch_size, max_question_token + max_option_token, duma_encoding_size)
			MHA_1 = self.multi_head_attention(queries=E_P_i, keys=E_QA_i, values=E_QA_i)	# MHA_1	: (batch_size, max_article_token, duma_encoding_size)
			MHA_2 = self.multi_head_attention(queries=E_QA_i, keys=E_P_i, values=E_P_i)		# MHA_2	: (batch_size, max_question_token + max_option_token, duma_encoding_size)
			if self.k > 1:
				# Stack k layers
				for _ in range(self.k - 1):
					MHA_1 = self.multi_head_attention(queries=MHA_1, keys=MHA_2, values=MHA_2)		# MHA_1	: (batch_size, max_article_token, duma_encoding_size)
					MHA_2 = self.multi_head_attention(queries=MHA_2, keys=MHA_1, values=MHA_1)		# MHA_2	: (batch_size, max_question_token + max_option_token, duma_encoding_size)
			O_i = self._fuse(x=MHA_1+torch.max(E_P, axis=1)[0], y=MHA_2+torch.max(E_QA, axis=1)[0])	# O_i	: (batch_size, ?)
			O_list.append(O_i.unsqueeze(1))
		O = torch.cat(O_list, axis=1)																# O		: (batch_size, N_CHOICES, ?)
		return O
	