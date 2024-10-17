# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn
# Implementation of DCMN+

import torch
from torch.nn import Module, Linear, CosineSimilarity, NLLLoss, functional as F

from src.modules.easy import load_model


class DCMN(Module):
	"""Forward propagation algorithm of DCMN+
	Reference: [DCMN+: Dual Co-Matching Network for Multi-choice Reading Comprehension](https://arxiv.org/abs/1908.11511)
	- Input:
	  - $P = \{P_1, ..., P_N\}, P_i \in \R^{d×p}$
	  - $Q \in \R^{d×q}$
	  - $A = \{A_1, ..., A_m\}, A_j \in \R^{d×a}$
	- Output:
	  - $L \in \R^m$
	- Example:
	  >>> args = load_args(Config=ModuleConfig)
	  >>> kwargs = {"train_batch_size"						: 8,
					"max_article_sentence"					: 4,
					"max_article_sentence_token"			: 32,
					"max_question_token"					: 16,
					"max_option_token"						: 24,
					"dcmn_scoring_method"					: "cosine",
					"dcmn_num_passage_sentence_selection"	: 2,
					"dcmn_pretrained_model"					: "albert-base-v1",
					"dcmn_encoding_size"					: 768}
	  >>> update_args(args, **kwargs)
	  >>> P_size = (args.train_batch_size * args.max_article_sentence, args.max_article_sentence_token)
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
	  >>> dcmn = DCMN(args=args)
	  >>> dcmn_output = dcmn.forward(**test_input)"""
	loss_function = NLLLoss()
	def __init__(self, args):
		super(DCMN, self).__init__()
		self.p = args.max_article_sentence_token
		self.q = args.max_question_token
		self.a = args.max_option_token
		self.N = args.max_article_sentence
		self.m = args.n_choices
		self.l = args.dcmn_encoding_size
		self.scoring_method = args.dcmn_scoring_method
		self.num_passage_sentence_selection = args.dcmn_num_passage_sentence_selection

		if args.load_pretrained_model_in_module:
			self.pretrained_model = load_model(
				model_path = MODEL_SUMMARY[args.dcmn_pretrained_model]["path"],
				device = args.pretrained_model_device,
			)
			self.pretrained_model.eval()
		else:
			self.pretrained_model = None
		if self.scoring_method == "cosine":
			self.Cosine = CosineSimilarity(dim=-1)
		elif self.scoring_method == "bilinear":
			self.W_1 = Linear(self.l, self.l, bias=False)
			self.W_2 = Linear(self.l, self.l, bias=False)
			self.W_3 = Linear(self.l, 1, bias=False)
			self.W_4 = Linear(self.l, 1, bias=False)
		else:
			raise Exception(f"Unknown scoring method: {self.scoring_method}")
		self.W_5 = Linear(self.l, self.l, bias=False)
		self.W_6 = Linear((self.m - 1) * self.l, self.l, bias=False)
		self.W_7 = Linear(self.l, self.l, bias=False)
		self.W_8 = Linear(self.l, self.l, bias=True)
		self.W_9 = Linear(self.l, self.l, bias=False)
		self.W_10 = Linear(self.l, self.l, bias=False)
		self.W_11 = Linear(self.l, self.l, bias=False)
		self.W_12 = Linear(self.l, self.l, bias=False)
		self.W_13 = Linear(self.l, self.l, bias=False)
		self.W_14 = Linear(self.l, self.l, bias=True)
		self.V = Linear(3 * self.l, 1, bias=False)
		
	# @param P		: {"input_ids": tensor, "token_type_ids": tensor, "attention_mask": tensor}, tensor(batch_size * max_article_sentence, max_article_sentence_token)
	# @param Q		: {"input_ids": tensor, "token_type_ids": tensor, "attention_mask": tensor}, tensor(batch_size, max_question_token)
	# @param A		: {"input_ids": tensor, "token_type_ids": tensor, "attention_mask": tensor}, tensor(batch_size * N_CHOICES, max_option_token)
	def forward(self, P, Q, A, pretrained_model=None):
		H_p, H_q, H_a = self.contextualized_encoding(P, Q, A, pretrained_model=pretrained_model)
		H_p_s = self.passage_sentence_selection(H_p, H_q, H_a)	# H_p_s			: (batch_size, dcmn_num_passage_sentence_selection * max_article_sentence_token, dcmn_encoding_size)
		H_o = self.answer_option_interaction(H_a)				# H_o			: (batch_size, N_CHOICES, max_option_token, dcmn_encoding_size)
		C = self.bidirectional_matching(H_p_s, H_q, H_o)		# C				: (batch_size, N_CHOICES, 3 * dcmn_encoding_size)
		L_unactived = self.V(C).squeeze(-1)						# L_unactived	: (batch_size, N_CHOICES)
		L = F.log_softmax(L_unactived, dim=-1)					# L				: (batch_size, N_CHOICES)
		return L

	# @param P		: {"input_ids": tensor, "token_type_ids": tensor, "attention_mask": tensor}, tensor(batch_size * max_article_sentence, max_article_sentence_token)
	# @param Q		: {"input_ids": tensor, "token_type_ids": tensor, "attention_mask": tensor}, tensor(batch_size, max_question_token)
	# @param A		: {"input_ids": tensor, "token_type_ids": tensor, "attention_mask": tensor}, tensor(batch_size * N_CHOICES, max_option_token)
	# @return H_p	: (batch_size * max_article_sentence, max_article_sentence_token, dcmn_encoding_size)
	# @return H_q	: (batch_size, max_question_token, dcmn_encoding_size)
	# @return H_a	: (batch_size * N_CHOICES, max_option_token, dcmn_encoding_size)
	def contextualized_encoding(self, P, Q, A, pretrained_model=None):
		if self.pretrained_model is None:
			from settings import DEVICE
			H_p = pretrained_model(**P).last_hidden_state[:, :, :self.l].to(DEVICE)	# H_p: (batch_size * max_article_sentence, max_article_sentence_token, dcmn_encoding_size)
			H_q = pretrained_model(**Q).last_hidden_state[:, :, :self.l].to(DEVICE)	# H_q: (batch_size, max_question_token, dcmn_encoding_size)
			H_a = pretrained_model(**A).last_hidden_state[:, :, :self.l].to(DEVICE)	# H_a: (batch_size * N_CHOICES, max_option_token, dcmn_encoding_size)
		else:
			H_p = self.pretrained_model(**P).last_hidden_state[:, :, :self.l]	# H_p: (batch_size * max_article_sentence, max_article_sentence_token, dcmn_encoding_size)
			H_q = self.pretrained_model(**Q).last_hidden_state[:, :, :self.l]	# H_q: (batch_size, max_question_token, dcmn_encoding_size)
			H_a = self.pretrained_model(**A).last_hidden_state[:, :, :self.l]	# H_a: (batch_size * N_CHOICES, max_option_token, dcmn_encoding_size)
		return H_p, H_q, H_a

	# @param H_p	: (batch_size * max_article_sentence, max_article_sentence_token, dcmn_encoding_size)
	# @param H_q	: (batch_size, max_question_token, dcmn_encoding_size)
	# @param H_a	: (batch_size * N_CHOICES, max_option_token, dcmn_encoding_size)
	# @return H_p_s	: (batch_size, dcmn_num_passage_sentence_selection * max_article_sentence_token, dcmn_encoding_size)
	def passage_sentence_selection(self, H_p, H_q, H_a):
		batch_size = H_q.size(0)
		H_p_decomposed = H_p.view(batch_size, self.N, self.p, self.l)		# H_p_decomposed: (batch_size, max_article_sentence, max_article_sentence_token, dcmn_encoding_size)
		if self.scoring_method == "cosine":
			H_a_decomposed = H_a.view(batch_size, self.m, self.a, self.l)	# H_a_decomposed: (batch_size, N_CHOICES, max_option_token, dcmn_encoding_size)
			H_p_decomposed_repeat_as_a = H_p_decomposed.repeat(1, self.m, self.a, 1)																# H_p_decomposed_repeat_as_a: (batch_size, N_CHOICES * max_article_sentence, max_option_token * max_article_sentence_token, dcmn_encoding_size)
			H_a_decomposed_repeat_as_p = H_a_decomposed.repeat(1, 1, self.N, self.p).view(batch_size, self.m * self.N, self.a * self.p, self.l)		# H_a_decomposed_repeat_as_p: (batch_size, N_CHOICES * max_article_sentence, max_option_token * max_article_sentence_token, dcmn_encoding_size)
			H_p_decomposed_repeat_as_q = H_p_decomposed.repeat(1, 1, self.q, 1)																		# H_p_decomposed_repeat_as_q: (batch_size, max_article_sentence, max_question_token * max_article_sentence_token, dcmn_encoding_size)
			H_q_repeat_as_p = H_q.unsqueeze(1).repeat(1, self.N, 1, self.p).view(batch_size, self.N, self.q * self.p, self.l)						# H_q_repeat_as_p: (batch_size, max_article_sentence, max_question_token * max_article_sentence_token, dcmn_encoding_size)
			D_pa = self.Cosine(H_p_decomposed_repeat_as_a, H_a_decomposed_repeat_as_p)					# D_pa		: (batch_size, N_CHOICES * max_article_sentence, max_option_token * max_article_sentence_token)
			D_pq = self.Cosine(H_p_decomposed_repeat_as_q, H_q_repeat_as_p)								# D_pq		: (batch_size, max_article_sentence, max_question_token * max_article_sentence_token)
			bar_D_pa = torch.max(D_pa.view(batch_size, self.m * self.N, self.a, self.p), axis=-1)[0]	# bar_D_pa	: (batch_size, N_CHOICES * max_article_sentence, max_option_token)
			bar_D_pq = torch.max(D_pq.view(batch_size, self.N, self.q, self.p), axis=-1)[0]				# bar_D_pq	: (batch_size, max_article_sentence, max_question_token)
			# Calculate score
			score_a = torch.mean(bar_D_pa, axis=-1)											# score_a	: (batch_size, N_CHOICES * max_article_sentence)
			score_q = torch.mean(bar_D_pq, axis=-1)											# score_q	: (batch_size, max_article_sentence)
			score = torch.sum(score_a.view(batch_size, self.m, self.N), axis=1) + score_q	# score		: (batch_size, max_article_sentence)
		elif self.scoring_method == "bilinear":				
			cache_1 = self.W_2(H_p)																					# cache_1				: (batch_size * max_article_sentence, max_article_sentence_token, dcmn_encoding_size)
			cache_1_repeat_as_a = cache_1.repeat(1, self.m, 1).view(batch_size * self.N * self.m, self.p, self.l)	# cache_1_repeat_as_a	: (batch_size * max_article_sentence * N_CHOICES, max_article_sentence_token, dcmn_encoding_size)
			# Calculate $\hat P^{pq}$
			alpha_q = F.softmax(self.W_1(H_q), dim=-1)											# alpha_q		: (batch_size, max_question_token, dcmn_encoding_size)
			q = torch.bmm(alpha_q.permute(0, 2, 1), H_q)										# q				: (batch_size, dcmn_encoding_size, dcmn_encoding_size)
			q_repeat_as_p = q.repeat(1, self.N, 1).view(batch_size * self.N, self.l, self.l)	# q_repeat_as_p	: (batch_size * max_article_sentence, dcmn_encoding_size, dcmn_encoding_size)
			bar_P_q = torch.bmm(cache_1, q_repeat_as_p)											# bar_P_q		: (batch_size * max_article_sentence, max_article_sentence_token, dcmn_encoding_size)
			hat_P_pq = torch.max(bar_P_q, axis=1)[0]											# hat_P_pq		: (batch_size * max_article_sentence, dcmn_encoding_size)
			# Calculate $\hat P^{pa}$
			alpha_a = F.softmax(self.W_1(H_a), dim=-1)																								# alpha_a		: (batch_size * N_CHOICES, max_option_token, dcmn_encoding_size)
			a = torch.bmm(alpha_a.permute(0, 2, 1), H_a)																							# a				: (batch_size * N_CHOICES, dcmn_encoding_size, dcmn_encoding_size)
			a_repeat_as_p = a.view(batch_size, self.m, self.l, self.l).repeat(1, self.N, 1, 1).view(batch_size * self.N * self.m, self.l, self.l)	# a_repeat_as_p	: (batch_size * max_article_sentence * N_CHOICES, dcmn_encoding_size, dcmn_encoding_size)
			bar_P_a = torch.bmm(cache_1_repeat_as_a, a_repeat_as_p)				# bar_P_a	: (batch_size * max_article_sentence * N_CHOICES, max_article_sentence_token, dcmn_encoding_size)
			hat_P_pa = torch.max(bar_P_a, axis=1)[0]							# hat_P_pq	: (batch_size * max_article_sentence * N_CHOICES, dcmn_encoding_size)
			# Calculate score
			score_q = self.W_3(hat_P_pq).view(batch_size, self.N)			# score_q	: (batch_size, max_article_sentence)
			score_a = self.W_4(hat_P_pa).view(batch_size, self.N, self.m)	# score_a	: (batch_size, max_article_sentence, N_CHOICES)
			score = score_q + torch.sum(score_a, axis=-1)					# score		: (batch_size, max_article_sentence)
		else:
			raise Exception(f"Unknown scoring method: {self.scoring_method}")
		# Sort in descending order by score and select sentences
		sorted_score_index = torch.sort(score, descending=True, axis=-1)[1]																																				# sorted_score_index: (batch_size, max_article_sentence)
		H_p_s = torch.stack([H_p_decomposed[i, sorted_score_index[i, :self.num_passage_sentence_selection], :, :] for i in range(batch_size)]).view(batch_size, self.num_passage_sentence_selection * self.p, self.l)	# H_p_s				: (batch_size, dcmn_num_passage_sentence_selection * max_article_sentence_token, dcmn_encoding_size)
		return H_p_s
	
	# @param H_a	: (batch_size * N_CHOICES, max_option_token, dcmn_encoding_size)
	# @return H_o	: (batch_size, N_CHOICES, max_option_token, dcmn_encoding_size)
	def answer_option_interaction(self, H_a):
		H_o_list = list()
		H_a_decomposed = H_a.view(-1, self.m, self.a, self.l)							# H_a_decomposed	: (batch_size, N_CHOICES_1, max_option_token, dcmn_encoding_size)
		for i in range(self.m):
			hat_H_a_i_list = list()
			H_a_i = H_a_decomposed[:, i, :, :]											# H_a_i				: (batch_size, max_option_token, dcmn_encoding_size)
			cache_1 = self.W_5(H_a_i)													# cache_1			: (batch_size, max_option_token, dcmn_encoding_size)
			for j in range(self.m):
				if not i == j:
					H_a_j = H_a_decomposed[:, j, :, :]									# H_a_j				: (batch_size, max_option_token, dcmn_encoding_size)
					G = F.softmax(torch.bmm(cache_1, H_a_j.permute(0, 2, 1)), dim=-1)	# G					: (batch_size, max_option_token, max_option_token)
					H_a_ij = F.relu(torch.bmm(G, H_a_j))								# H_a_ij			: (batch_size, max_option_token, dcmn_encoding_size)
					hat_H_a_i_list.append(H_a_ij)		
			hat_H_a_i = torch.cat(hat_H_a_i_list, axis=-1)								# hat_H_a_i			: (batch_size, max_option_token, (N_CHOICES - 1) * dcmn_encoding_size)
			bar_H_a_i = self.W_6(hat_H_a_i)												# bar_H_a_i			: (batch_size, max_option_token, dcmn_encoding_size)
			g = torch.sigmoid(self.W_7(bar_H_a_i) + self.W_8(bar_H_a_i))				# g					: (batch_size, max_option_token, dcmn_encoding_size)
			H_o_i = g * H_a_i + (1 - g) * bar_H_a_i										# H_o_i				: (batch_size, max_option_token, dcmn_encoding_size)
			H_o_list.append(H_o_i.unsqueeze(1))
		H_o = torch.cat(H_o_list, axis=1).view(-1, self.m, self.a, self.l)				# H_o				: (batch_size, N_CHOICES, max_option_token, dcmn_encoding_size)
		return H_o

	# @param H_p_s	: (batch_size, dcmn_num_passage_sentence_selection * max_article_sentence_token, dcmn_encoding_size)
	# @param H_q	: (batch_size, max_question_token, dcmn_encoding_size)
	# @param H_o	: (batch_size, N_CHOICES, max_option_token, dcmn_encoding_size)
	# @return C		: (batch_size, N_CHOICES, 3 * dcmn_encoding_size)
	def bidirectional_matching(self, H_p_s, H_q, H_o):
		cache_1 = self.W_9(H_p_s)															# cache_1	: (batch_size, dcmn_num_passage_sentence_selection * max_article_sentence_token, dcmn_encoding_size)
		cache_2 = self.W_9(H_q)																# cache_2	: (batch_size, max_question_token, dcmn_encoding_size)
		cache_3 = self.W_10(H_q)															# cache_3	: (batch_size, max_question_token, dcmn_contextualized_encoding)
		M_pq = self._matching_function(H_p_s, H_q, G_xy_left=cache_1, G_yx_left=cache_3)	# M_pq		: (batch_size, dcmn_encoding_size)
		C_list = list()
		for i in range(self.m):
			H_o_i = H_o[:, i, :, :]																		# H_o_i		: (batch_size, max_option_token, dcmn_encoding_size)
			cache_4 = self.W_10(H_o_i)																	# cache_4	: (batch_size, max_option_token, dcmn_encoding_size)
			M_po_i = self._matching_function(H_x=H_p_s, H_y=H_o_i, G_xy_left=cache_1, G_yx_left=cache_4)# M_po		: (batch_size, dcmn_encoding_size)
			M_qo_i = self._matching_function(H_x=H_q, H_y=H_o_i, G_xy_left=cache_2, G_yx_left=cache_4)	# M_qo		: (batch_size, dcmn_encoding_size)
			C_i = torch.cat([M_pq, M_po_i, M_qo_i], axis=-1)											# C_i		: (batch_size, 3 * dcmn_encoding_size)
			C_list.append(C_i.unsqueeze(1))
		C = torch.cat(C_list, axis=1)																	# C			: (batch_size, N_CHOICES, 3 * dcmn_encoding_size)
		return C

	# @param H_x		: (batch_size, sequence_length_x, dcmn_encoding_size)
	# @param H_y		: (batch_size, sequence_length_y, dcmn_encoding_size)
	# @param G_xy_left	: (batch_size, sequence_length_x, dcmn_encoding_size), cache for self.W_9(H_x)
	# @param G_yx_left	: (batch_size, sequence_length_y, dcmn_encoding_size), cache for self.W_10(H_y)
	# @return M_xy		: (batch_size, dcmn_encoding_size)
	def _matching_function(self, H_x, H_y, G_xy_left=None, G_yx_left=None):
		G_xy = F.softmax(torch.bmm(self.W_9(H_x) if G_xy_left is None else G_xy_left, H_y.permute(0, 2, 1)), dim=-1)	# G_xy	: (batch_size, sequence_length_x, sequence_length_y)
		G_yx = F.softmax(torch.bmm(self.W_10(H_y) if G_yx_left is None else G_yx_left, H_x.permute(0, 2, 1)), dim=-1)	# G_yx	: (batch_size, sequence_length_y, sequence_length_x)
		E_x = torch.bmm(G_xy, H_y)																						# E_x	: (batch_size, sequence_length_x, dcmn_encoding_size)
		E_y = torch.bmm(G_yx, H_x)																						# E_y	: (batch_size, sequence_length_y, dcmn_encoding_size)
		S_x = F.relu(self.W_11(E_x))																					# S_x	: (batch_size, sequence_length_x, dcmn_encoding_size)
		S_y = F.relu(self.W_12(E_y))																					# S_y	: (batch_size, sequence_length_y, dcmn_encoding_size)
		S_xy = torch.max(S_x, axis=1)[0]																				# S_xy	: (batch_size, dcmn_encoding_size)
		S_yx = torch.max(S_y, axis=1)[0]																				# S_yx	: (batch_size, dcmn_encoding_size)
		g = torch.sigmoid(self.W_13(S_xy) + self.W_14(S_yx))															# g		: (batch_size, dcmn_encoding_size)
		M_xy = g * S_xy + (1 - g) * S_yx																				# M_xy	: (batch_size, dcmn_encoding_size)
		return M_xy
