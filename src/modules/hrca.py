# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn
# Implementation of HRCA

import torch
from torch.nn import Module, Linear, NLLLoss, functional as F

from settings import DEVICE
from src.modules.easy import load_model
from src.modules.attention import MultiHeadAttention



class HRCA(Module):
	"""Forward propagation algorithm of HRCA and HRCA+
	Reference: [HRCA+: Advanced Multiple-choice Machine Reading Comprehension Method](www.lrec-conf.org/proceedings/lrec2022/pdf/2022.lrec-1.651.pdf)
	- Input:
	  - $P \in \R^{d×p}$ 
	  - $Q \in \R^{d×q}$
	  - $A = \{A_1, ..., A_m\}, A_j \in \R^{d×a}$
	- Output:
	  - $L \in \R^m$"""
	loss_function = NLLLoss()
	def __init__(self, args):
		super(HRCA, self).__init__()
		self.device = args.device
		self.p = args.max_article_token
		self.q = args.max_question_token
		self.a = args.max_option_token
		self.m = args.n_choices
		self.l = args.hrca_encoding_size
		self.k = args.hrca_num_layers
		self.fuse_method = args.hrca_fuse_method
		self.plus = args.hrca_plus

		self.multi_head_attention = MultiHeadAttention(d_model=args.hrca_encoding_size, num_heads=args.hrca_mha_num_heads, dropout_rate=args.hrca_mha_dropout_rate)
		self.fuse_linear_x = Linear(self.l, self.l, bias=True)
		self.fuse_linear_y = Linear(self.l, self.l, bias=True)
		self.fuse_linear_z = Linear(self.l, self.l, bias=True)
		if self.fuse_method in ["mul", "sum"]:
			self.W = Linear(self.l, 1, bias=False)
		elif self.fuse_method == "cat":
			self.W = Linear(3 * self.l, 1, bias=False)
		else:
			raise Exception(f"Unknown fuse method: {self.fuse_method}")
		if args.load_pretrained_model_in_module:
			self.pretrained_model = load_model(
				model_path = MODEL_SUMMARY[args.hrca_pretrained_model]["path"],
				device = args.pretrained_model_device,
			)
			self.pretrained_model.eval()
		else:
			self.pretrained_model = None
	# @param P	: {"input_ids": tensor, "token_type_ids": tensor, "attention_mask": tensor}, tensor(batch_size, max_article_token)
	# @param Q	: {"input_ids": tensor, "token_type_ids": tensor, "attention_mask": tensor}, tensor(batch_size, max_question_token)
	# @param A	: {"input_ids": tensor, "token_type_ids": tensor, "attention_mask": tensor}, tensor(batch_size * N_CHOICES, max_option_token)
	def forward(self, P, Q, A, pretrained_model=None):
		E_P, E_Q, E_A = self.contextualized_encoding(P, Q, A, pretrained_model=pretrained_model)
		O = self.human_reading_comprehension_attention(E_P, E_Q, E_A)	# O				: (batch_size, N_CHOICES, ?)
		L_unactived = self.W(O).squeeze(-1)								# L_unactived	: (batch_size, N_CHOICES)
		L = F.log_softmax(L_unactived, dim=-1)							# L				: (batch_size, N_CHOICES)
		return L

	# @param P		: {"input_ids": tensor, "token_type_ids": tensor, "attention_mask": tensor}, tensor(batch_size, max_article_token)
	# @param Q		: {"input_ids": tensor, "token_type_ids": tensor, "attention_mask": tensor}, tensor(batch_size, max_question_token)
	# @param A		: {"input_ids": tensor, "token_type_ids": tensor, "attention_mask": tensor}, tensor(batch_size * N_CHOICES, max_option_token)
	# @return E_P	: (batch_size, N_CHOICES, max_article_token, hrca_encoding_size)
	# @return E_Q	: (batch_size, N_CHOICES, max_question_token, hrca_encoding_size)
	# @return E_A	: (batch_size, N_CHOICES, max_option_token, hrca_encoding_size)
	def contextualized_encoding(self, P, Q, A, pretrained_model=None):
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
		E = torch.cat(E_list, axis=1)			# E		: (batch_size, N_CHOICES, max_article_token + max_question_token + max_option_token, hrca_encoding_size)
		E_P = E[:, :, :self.p, :]				# E_P	: (batch_size, N_CHOICES, max_article_token, hrca_encoding_size)
		E_Q = E[:, :, self.p:self.p+self.q, :]	# E_Q	: (batch_size, N_CHOICES, max_question_token, hrca_encoding_size)
		E_A = E[:, :, self.p+self.q:, :]		# E_A	: (batch_size, N_CHOICES, max_option_token, hrca_encoding_size)
		return E_P.to(self.device), E_Q.to(self.device), E_A.to(self.device)

	# @param E_P: (batch_size, N_CHOICES, max_article_token, hrca_encoding_size)
	# @param E_Q: (batch_size, N_CHOICES, max_question_token, hrca_encoding_size)
	# @param E_A: (batch_size, N_CHOICES, max_option_token, hrca_encoding_size)
	# @return O	: (batch_size, N_CHOICES, ?) where ? could be hrca_encoding_size or 3 * hrca_encoding_size
	def human_reading_comprehension_attention(self, E_P, E_Q, E_A):
		O_list = list()
		for i in range(self.m):
			E_P_U, E_Q_U, E_A_U = self._hrca(E_P[:, i, :, :], E_Q[:, i, :, :], E_A[:, i, :, :])
			if self.k > 1:
				# Stack k layers
				for _ in range(self.k - 1):
					E_P_U, E_Q_U, E_A_U = self._hrca(E_P_U, E_Q_U, E_A_U)
			O_i = self._fuse(E_P_U, E_Q_U, E_A_U)	# O_i	: (batch_size, ?)
			O_list.append(O_i.unsqueeze(1))
		O = torch.cat(O_list, axis=1)				# O		: (batch_size, N_CHOICES, ?)
		return O

	# @param E_P_U	: (batch_size, max_article_token, hrca_encoding_size)
	# @param E_Q_U	: (batch_size, max_question_token, hrca_encoding_size)
	# @param E_A_U	: (batch_size, max_option_token, hrca_encoding_size)
	# @return E_P_U	: (batch_size, max_article_token, hrca_encoding_size)
	# @return E_Q_U	: (batch_size, max_question_token, hrca_encoding_size)
	# @return E_A_U	: (batch_size, max_option_token, hrca_encoding_size)
	def _hrca(self, E_P_U, E_Q_U, E_A_U):
		if self.plus:
			# HRCA: Q2Q -> O2Q -> P2O
			E_Q_U = self.multi_head_attention(queries=E_Q_U, keys=E_Q_U, values=E_Q_U)	# E_Q_U: (batch_size, max_question_token, hrca_encoding_size)
			E_A_U = self.multi_head_attention(queries=E_A_U, keys=E_Q_U, values=E_Q_U)	# E_A_U: (batch_size, max_option_token, hrca_encoding_size)
			E_P_U = self.multi_head_attention(queries=E_P_U, keys=E_A_U, values=E_A_U)	# E_P_U: (batch_size, max_article_token, hrca_encoding_size)
		else:
			# HRCA+: Q2Q -> Q2O -> O2O -> O2Q -> O2P -> Q2P -> P2P -> P2Q -> P2O
			E_Q_U = self.multi_head_attention(queries=E_Q_U, keys=E_Q_U, values=E_Q_U)	# E_Q_U: (batch_size, max_question_token, hrca_encoding_size)
			E_Q_U = self.multi_head_attention(queries=E_Q_U, keys=E_A_U, values=E_A_U)	# E_Q_U: (batch_size, max_question_token, hrca_encoding_size)
			E_A_U = self.multi_head_attention(queries=E_A_U, keys=E_A_U, values=E_A_U)	# E_A_U: (batch_size, max_option_token, hrca_encoding_size)
			E_A_U = self.multi_head_attention(queries=E_A_U, keys=E_Q_U, values=E_Q_U)	# E_A_U: (batch_size, max_option_token, hrca_encoding_size)
			E_A_U = self.multi_head_attention(queries=E_A_U, keys=E_P_U, values=E_P_U)	# E_A_U: (batch_size, max_option_token, hrca_encoding_size)
			E_Q_U = self.multi_head_attention(queries=E_Q_U, keys=E_P_U, values=E_P_U)	# E_Q_U: (batch_size, max_question_token, hrca_encoding_size)
			E_P_U = self.multi_head_attention(queries=E_P_U, keys=E_P_U, values=E_P_U)	# E_P_U: (batch_size, max_article_token, hrca_encoding_size)
			E_P_U = self.multi_head_attention(queries=E_P_U, keys=E_Q_U, values=E_Q_U)	# E_P_U: (batch_size, max_article_token, hrca_encoding_size)
			E_P_U = self.multi_head_attention(queries=E_P_U, keys=E_A_U, values=E_A_U)	# E_P_U: (batch_size, max_article_token, hrca_encoding_size)			
		return E_P_U, E_Q_U, E_A_U
	
	# @param x	: (batch_size, x_length, hrca_encoding_size)
	# @param y	: (batch_size, y_length, hrca_encoding_size)
	# @param z	: (batch_size, z_length, hrca_encoding_size)
	# @return	: (batch_size, ?) where ? could be hrca_encoding_size or 3 * hrca_encoding_size
	def _fuse(self, x, y, z):
		x_project = self.fuse_linear_x(x)					# x_project			: (batch_size, x_length, hrca_encoding_size)
		y_project = self.fuse_linear_y(y)					# y_project			: (batch_size, y_length, hrca_encoding_size)
		z_project = self.fuse_linear_z(z)					# z_project			: (batch_size, z_length, hrca_encoding_size)
		x_project_pooled = torch.max(x_project, axis=1)[0]	# x_project_pooled	: (batch_size, hrca_encoding_size)
		y_project_pooled = torch.max(y_project, axis=1)[0]	# y_project_pooled	: (batch_size, hrca_encoding_size)
		z_project_pooled = torch.max(z_project, axis=1)[0]	# z_project_pooled	: (batch_size, hrca_encoding_size)
		if self.fuse_method == "mul":
			return torch.sigmoid(x_project_pooled * y_project_pooled * z_project_pooled)					# @return	: (batch_size, hrca_encoding_size)
		elif self.fuse_method == "sum":
			return torch.sigmoid(x_project_pooled + y_project_pooled + z_project_pooled)					# @return	: (batch_size, hrca_encoding_size)
		elif self.fuse_method == "cat":
			return torch.sigmoid(torch.cat([x_project_pooled, y_project_pooled, z_project_pooled], axis=-1))# @return	: (batch_size, 3 * hrca_encoding_size)
		else:
			raise Exception(f"Unknown fuse method: {self.fuse_method}")

class HRCAv1(HRCA):
	"""Change the encoder algorithm of HRCA"""
	def __init__(self, args):
		super(HRCAv1, self).__init__(args)

	# @param P		: {"input_ids": tensor, "token_type_ids": tensor, "attention_mask": tensor}, tensor(batch_size, max_article_token)
	# @param Q		: {"input_ids": tensor, "token_type_ids": tensor, "attention_mask": tensor}, tensor(batch_size, max_question_token)
	# @param A		: {"input_ids": tensor, "token_type_ids": tensor, "attention_mask": tensor}, tensor(batch_size * N_CHOICES, max_option_token)
	# @return E_P	: (batch_size, N_CHOICES, max_article_token, hrca_encoding_size)
	# @return E_Q	: (batch_size, N_CHOICES, max_question_token, hrca_encoding_size)
	# @return E_A	: (batch_size, N_CHOICES, max_option_token, hrca_encoding_size)
	def contextualized_encoding(self, P, Q, A, pretrained_model=None):
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
		E_QA = torch.cat(E_QA_list, axis=1)																										# E_QA			: (batch_size, N_CHOICES, max_question_token + max_option_token, hrca_encoding_size)
		E_Q = E_QA[:, :, :self.q, :]																											# E_Q			: (batch_size, N_CHOICES, max_question_token, hrca_encoding_size)
		E_A = E_QA[:, :, self.q:, :]																											# E_A			: (batch_size, N_CHOICES, max_option_token, hrca_encoding_size)
		E_P_unrepeated = pretrained_model(**P).last_hidden_state if self.pretrained_model is None else pretrained_model(**P).last_hidden_state	# E_P_unrepeated: (batch_size, max_article_token, duma_encoding_size)
		E_P = E_P_unrepeated.unsqueeze(1).repeat(1, self.m, 1, 1)																				# E_P			: (batch_size, N_CHOICES, max_article_token, duma_encoding_size)
		return E_P.to(DEVICE), E_Q.to(DEVICE), E_A.to(DEVICE)