# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn
# Implementation of Co-Matching

import torch
from torch.nn import Module, Linear, LSTM, Dropout, NLLLoss, functional as F

from src.modules.easy import MaskedLSTM


class CoMatchBranch(Module):
	"""Forward propagation algorithm of one branch in Co-Matching
	Reference: [A Co-Matching Model for Multi-choice Reading Comprehension](https://arxiv.org/abs/1806.04068)
	- Input: 
	  - $P_i \in \R^{d×p}$
	  - $Q \in \R^{d×q}$
	  - $A_j \in \R^{d×a}$
	  where $d$ is embedding size, $p, q, a$ are namely the sequence length of passage, question and answer.
	- Output:
	  - $h^s_i \in \R^{l}$
	- Example:
	  >>> batch_size = 32
	  >>> args = load_args(Config=ModuleConfig)
	  >>> comatch_branch = CoMatchBranch(args=args)
	  >>> P = torch.FloatTensor(batch_size, args.max_article_sentence_token, args.comatch_embedding_size)
	  >>> Q = torch.FloatTensor(batch_size, args.max_question_token, args.comatch_embedding_size)
	  >>> A = torch.FloatTensor(batch_size, args.max_option_token, args.comatch_embedding_size)
	  >>> P_shape = torch.LongTensor(batch_size * [80])
	  >>> Q_shape = torch.LongTensor(batch_size * [50])
	  >>> A_shape = torch.LongTensor(batch_size * [60])
	  >>> h_s = comatch_branch(P, Q, A)"""
	def __init__(self, args):
		super(CoMatchBranch, self).__init__()
		self.p = args.max_article_sentence_token
		self.q = args.max_question_token
		self.a = args.max_option_token
		self.d = args.comatch_embedding_size
		self.l = args.comatch_bilstm_hidden_size

		self.Encoder_P = MaskedLSTM(input_size		= self.d,
									hidden_size		= int(self.l / (1 + args.comatch_bilstm_bidirectional)),
									num_layers		= args.comatch_bilstm_num_layers,
									batch_first		= True,
									bidirectional	= args.comatch_bilstm_bidirectional,
									dropout			= args.comatch_bilstm_dropout)
		self.Encoder_Q = MaskedLSTM(input_size		= self.d,
									hidden_size		= int(self.l / (1 + args.comatch_bilstm_bidirectional)),
									num_layers		= args.comatch_bilstm_num_layers,
									batch_first		= True,
									bidirectional	= args.comatch_bilstm_bidirectional,
									dropout			= args.comatch_bilstm_dropout)
		self.Encoder_A = MaskedLSTM(input_size		= self.d,
									hidden_size		= int(self.l / (1 + args.comatch_bilstm_bidirectional)),
									num_layers		= args.comatch_bilstm_num_layers,
									batch_first		= True,
									bidirectional	= args.comatch_bilstm_bidirectional,
									dropout			= args.comatch_bilstm_dropout)
		self.Encoder_C = LSTM(input_size	= 2 * self.l,
							  hidden_size	= int(self.l / (1 + args.comatch_bilstm_bidirectional)),
							  num_layers	= args.comatch_bilstm_num_layers,
							  batch_first	= True,
							  bidirectional	= args.comatch_bilstm_bidirectional,
							  dropout		= 0.)
		self.W_g = Linear(self.l, self.l, bias=True)
		self.W_m = Linear(2 * self.l, self.l, bias=True)

	# @param P		: (batch_size, max_article_sentence_token, embedding_size)
	# @param Q		: (batch_size, max_question_token, embedding_size)
	# @param A		: (batch_size, max_option_token, embedding_size)
	# @param P_shape: (batch_size, )
	# @param Q_shape: (batch_size, )
	# @param A_shape: (batch_size, )
	def forward(self, P, Q, A, P_shape, Q_shape, A_shape):
		H_p = self.Encoder_P(P, P_shape)													# H_p				: (batch_size, max_article_sentence_token, comatch_bilstm_hidden_size)
		H_q = self.Encoder_Q(Q, Q_shape)													# H_q				: (batch_size, max_question_token, comatch_bilstm_hidden_size)
		H_a = self.Encoder_A(A, A_shape)													# H_a				: (batch_size, max_option_token, comatch_bilstm_hidden_size)
		H_p_T = H_p.permute(0, 2, 1)														# H_p_T				: (batch_size, comatch_bilstm_hidden_size, max_article_sentence_token)
		H_q_T = H_q.permute(0, 2, 1)														# H_q_T				: (batch_size, comatch_bilstm_hidden_size, max_question_token)
		H_a_T = H_a.permute(0, 2, 1)														# H_a_T				: (batch_size, comatch_bilstm_hidden_size, max_option_token)
		G_q = F.softmax(torch.bmm(self.W_g(H_q), H_p_T), dim=-1)							# G_q				: (batch_size, max_question_token, max_article_sentence_token)
		G_a = F.softmax(torch.bmm(self.W_g(H_a), H_p_T), dim=-1)							# G_a				: (batch_size, max_option_token, max_article_sentence_token)
		bar_H_q = torch.bmm(H_q_T, G_q)														# bar_H_q			: (batch_size, comatch_bilstm_hidden_size, max_article_sentence_token)
		bar_H_a = torch.bmm(H_a_T, G_a)														# bar_H_a			: (batch_size, comatch_bilstm_hidden_size, max_article_sentence_token)
		bar_H_q_T = bar_H_q.permute(0, 2, 1)												# bar_H_q_T			: (batch_size, max_article_sentence_token, comatch_bilstm_hidden_size)
		bar_H_a_T = bar_H_a.permute(0, 2, 1)												# bar_H_a_T			: (batch_size, max_article_sentence_token, comatch_bilstm_hidden_size)
		M_q = F.relu(self.W_m(torch.cat([bar_H_q_T - H_p, bar_H_q_T * H_p], axis=-1)))		# M_q				: (batch_size, max_article_sentence_token, comatch_bilstm_hidden_size)
		M_a = F.relu(self.W_m(torch.cat([bar_H_a_T - H_p, bar_H_a_T * H_p], axis=-1)))		# M_a				: (batch_size, max_article_sentence_token, comatch_bilstm_hidden_size)
		C = torch.cat([M_q, M_a], axis=-1)													# C					: (batch_size, max_article_sentence_token, 2 * comatch_bilstm_hidden_size)
		h_s_unpooled, _ = self.Encoder_C(C)													# h_s_unpooled		: (batch_size, max_article_sentence_token, comatch_bilstm_hidden_size)
		h_s_unsqueezed = F.max_pool1d(h_s_unpooled.permute(0, 2, 1), kernel_size=self.p)	# h_s_unsqueezed	: (batch_size, comatch_bilstm_hidden_size, 1)
		h_s = h_s_unsqueezed.squeeze(-1)													# h_s				: (batch_size, comatch_bilstm_hidden_size)
		return h_s


class VerbosedCoMatch(Module):
	"""Forward propagation algorithm of Co-Matching
	Reference: [A Co-Matching Model for Multi-choice Reading Comprehension](https://arxiv.org/abs/1806.04068)
	- Input:
	  - $P = \{P_1, ..., P_N\}, P_i \in \R^{d×p}$
	  - $Q \in \R^{d×q}$
	  - $A = \{A_1, ..., A_m\}, A_j \in \R^{d×a}$
	  where $N$ is the number of sentences in article, $d$ is embedding size, $p, q, a$ are namely the sequence length of passage, question and answer.   
	- Output:
	  - $L \in \R^m$
	- Example:
	  >>> batch_size = 32
	  >>> args = load_args(Config=ModuleConfig)
	  >>> test_input = {'P'			: [torch.FloatTensor(batch_size, args.max_article_sentence_token, args.comatch_embedding_size) for _ in range(args.max_article_sentence, )],
						'Q'			: torch.FloatTensor(batch_size, args.max_question_token, args.comatch_embedding_size),
						'A'			: [torch.FloatTensor(batch_size, args.max_option_token, args.comatch_embedding_size) for _ in range(N_CHOICES)],
						'P_shape'	: [torch.LongTensor([100] * batch_size) for _ in range(args.max_article_sentence)],
						'Q_shape'	: torch.LongTensor([100] * batch_size),
						'A_shape'	: [torch.LongTensor([100] * batch_size) for _ in range(N_CHOICES)],
						}
	  >>> verbosed_comatch = VerbosedCoMatch(args=args)
	  >>> logit_output = verbosed_comatch(**test_input)
	- It takes about 1 minute to run the forward function for standard input size. TOO SLOW!
	"""
	def __init__(self, args):
		super(VerbosedCoMatch, self).__init__()
		self.l = args.comatch_bilstm_hidden_size
		self.N = args.max_article_sentence
		self.comatch_branch = CoMatchBranch(args=args)
		self.Encoder_H_s = LSTM(input_size	= self.l,
								hidden_size	= int(self.l / (1 + args.comatch_bilstm_bidirectional)),
								num_layers	= args.comatch_bilstm_num_layers,
								batch_first	= True,
								bidirectional	= args.comatch_bilstm_bidirectional)
		self.w = Linear(self.l, 1, bias=False)

	# @param P		: List[FloatTensor] max_article_sentence × (batch_size, max_article_sentence_token, embedding_size)
	# @param Q		: (batch_size, max_question_token, embedding_size)
	# @param A		: List[FloatTensor] N_CHOICES × (batch_size, max_option_token, embedding_size)
	# @param P_shape: List[LongTensor] max_article_sentence × (batch_size, )
	# @param Q_shape: (batch_size, )
	# @param A_shape: List[LongTensor] N_CHOICES × (batch_size, )
	def forward(self, P, Q, A, P_shape, Q_shape, A_shape):
		assert len(P) == len(P_shape)
		assert len(A) == len(A_shape)
		L_unactived = list()
		for (A_i, A_i_shape) in zip(A, A_shape):
			H_s = torch.cat([self.comatch_branch(P_i, Q, A_i,
												 P_i_shape,
												 Q_shape,
												 A_i_shape).unsqueeze(1) for (P_i, P_i_shape) in zip(P, P_shape)], axis=1)	# H_s				: (batch_size, max_article_sentence, comatch_bilstm_hidden_size)
			h_t_i_unpooled, _ = self.Encoder_H_s(H_s)																		# h_t_i_unpooled	: (batch_size, max_article_sentence, comatch_bilstm_hidden_size)
			h_t_i_unsqueezed = F.max_pool1d(h_t_i_unpooled.permute(0, 2, 1), kernel_size=self.N)							# h_t_i_unsqueezed	: (batch_size, comatch_bilstm_hidden_size, 1)
			h_t_i = h_t_i_unsqueezed.squeeze(-1)																			# h_t_i				: (batch_size, comatch_bilstm_hidden_size)
			L_unactived.append(self.w(h_t_i))																				# self.w(h_t_i)		: (batch_size, 1)
		L_unactived = torch.cat(L_unactived, axis=-1)																		# L_unactived		: (batch_size, N_CHOICES)
		L = F.log_softmax(L_unactived, dim=-1)																				# L					: (batch_size, N_CHOICES)
		return L


class CoMatch(Module):
	"""Forward propagation algorithm of Co-Matching
	Reference: [A Co-Matching Model for Multi-choice Reading Comprehension](https://arxiv.org/abs/1806.04068)
	- Input:
	  - $P = \{P_1, ..., P_N\}, P_i \in \R^{d×p}$
	  - $Q \in \R^{d×q}$
	  - $A = \{A_1, ..., A_m\}, A_j \in \R^{d×a}$
	  where $N$ is the number of sentences in article, $d$ is embedding size, $p, q, a$ are namely the sequence length of passage, question and answer.   
	- Output:
	  - $L \in \R^m$
	- Example:
	  >>> batch_size = 32
	  >>> args = load_args(Config=ModuleConfig)
	  >>> test_input = {'P'			: torch.FloatTensor(batch_size, args.max_article_sentence, args.max_article_sentence_token, args.comatch_embedding_size),
						'Q'			: torch.FloatTensor(batch_size, args.max_question_token, args.comatch_embedding_size),
						'A'			: torch.FloatTensor(batch_size, N_CHOICES, args.max_option_token, args.comatch_embedding_size),
						'P_shape'	: torch.LongTensor([[100] * args.max_article_sentence] * batch_size),
						'Q_shape'	: torch.LongTensor([100] * batch_size),
						'A_shape'	: torch.LongTensor([[100] * N_CHOICES] * batch_size)}
	  >>> comatch = CoMatch(args=args)
	  >>> logit_output = comatch(**test_input)
	- It takes about 2 seconds to run the forward function for standard input size. GOOD DEAL!
	"""
	loss_function = NLLLoss()
	def __init__(self, args):
		super(CoMatch, self).__init__()
		self.N = args.max_article_sentence
		self.p = args.max_article_sentence_token
		self.q = args.max_question_token
		self.a = args.max_option_token
		self.d = args.comatch_embedding_size
		self.l = args.comatch_bilstm_hidden_size
		self.Encoder_P = MaskedLSTM(input_size		= self.d,
									hidden_size		= int(self.l / (1 + args.comatch_bilstm_bidirectional)),
									num_layers		= args.comatch_bilstm_num_layers,
									batch_first		= True,
									bidirectional	= args.comatch_bilstm_bidirectional,
									dropout			= args.comatch_bilstm_dropout)
		self.Encoder_Q = MaskedLSTM(input_size		= self.d,
									hidden_size		= int(self.l / (1 + args.comatch_bilstm_bidirectional)),
									num_layers		= args.comatch_bilstm_num_layers,
									batch_first		= True,
									bidirectional	= args.comatch_bilstm_bidirectional,
									dropout			= args.comatch_bilstm_dropout)
		self.Encoder_A = MaskedLSTM(input_size		= self.d,
									hidden_size		= int(self.l / (1 + args.comatch_bilstm_bidirectional)),
									num_layers		= args.comatch_bilstm_num_layers,
									batch_first		= True,
									bidirectional	= args.comatch_bilstm_bidirectional,
									dropout			= args.comatch_bilstm_dropout)
		self.Encoder_C = LSTM(input_size	= 2 * self.l,
							  hidden_size	= int(self.l / (1 + args.comatch_bilstm_bidirectional)),
							  num_layers	= args.comatch_bilstm_num_layers,
							  batch_first	= True,
							  bidirectional	= args.comatch_bilstm_bidirectional,
							  dropout		= 0.)
		self.Encoder_H_s = LSTM(input_size	= self.l,
								hidden_size	= int(self.l / (1 + args.comatch_bilstm_bidirectional)),
								num_layers	= args.comatch_bilstm_num_layers,
								batch_first	= True,
								bidirectional	= args.comatch_bilstm_bidirectional)
		self.W_g = Linear(self.l, self.l, bias=True)
		self.W_m = Linear(2 * self.l, self.l, bias=True)
		self.w = Linear(self.l, 1, bias=False)

	# @param P		: (batch_size, max_article_sentence, max_article_sentence_token, embedding_size)
	# @param Q		: (batch_size, max_question_token, embedding_size)
	# @param A		: (batch_size, N_CHOICES, max_option_token, embedding_size)
	# @param P_shape: (batch_size, max_article_sentence)
	# @param Q_shape: (batch_size, )
	# @param A_shape: (batch_size, N_CHOICES)
	# As below, when repeating P.size(1), we simply repeat(P.size(1), 1, 1), while repeating A.size(1), we repeat(1, A.size(1), 1).view(-1, .size(1), .size(2))
	# In this way, the first `max_article_sentence` rows refer to the sentence matching with choice A, the second `max_article_sentence` rows refer to that of choice B, as so on.
	def forward(self, P, Q, A, P_shape, Q_shape, A_shape, **kwargs):
		H_p = self.Encoder_P(P.view(-1, self.p, self.d), P_shape.view(-1, ))																# H_p						: (batch_size * max_article_sentence, max_article_sentence_token, comatch_bilstm_hidden_size)
		H_q = self.Encoder_Q(Q, Q_shape)																									# H_q						: (batch_size, max_question_token, comatch_bilstm_hidden_size)
		H_a = self.Encoder_A(A.view(-1, self.a, self.d), A_shape.view(-1, ))																# H_a						: (batch_size * N_CHOICES, max_option_token, comatch_bilstm_hidden_size)
		H_p_T = H_p.permute(0, 2, 1)																										# H_p_T						: (batch_size * max_article_sentence, comatch_bilstm_hidden_size, max_article_sentence_token)
		H_q_T = H_q.permute(0, 2, 1)																										# H_q_T						: (batch_size, comatch_bilstm_hidden_size, max_question_token)
		H_a_T = H_a.permute(0, 2, 1)																										# H_a_T						: (batch_size * N_CHOICES, comatch_bilstm_hidden_size, max_option_token)
		H_p_repeat_as_a = H_p.view(-1, P.size(1), H_p.size(1), H_p.size(2)).repeat(1, A.size(1), 1, 1).view(-1, H_p.size(1), H_p.size(2))	# H_p_repeat_as_a			: (batch_size * N_CHOICES * max_article_sentence, max_article_sentence_token, comatch_bilstm_hidden_size)
		H_p_T_repeat_as_a = H_p_repeat_as_a.permute(0, 2, 1)																				# H_p_T_repeat_as_a			: (batch_size * N_CHOICES * max_article_sentence, comatch_bilstm_hidden_size, max_article_sentence_token)
		G_q = F.softmax(torch.bmm(self.W_g(H_q).repeat(1, P.size(1), 1).view(-1, H_q.size(1), self.l), H_p_T), dim=-1)						# G_q						: (batch_size * max_article_sentence, max_question_token, max_article_sentence_token)
		G_a = F.softmax(torch.bmm(self.W_g(H_a).repeat(1, P.size(1), 1).view(-1, H_a.size(1), self.l), H_p_T_repeat_as_a), dim=-1)			# G_a						: (batch_size * N_CHOICES * max_article_sentence, max_option_token, max_article_sentence_token)
		bar_H_q = torch.bmm(H_q_T.repeat(1, P.size(1), 1).view(-1, H_q_T.size(1), H_q_T.size(2)), G_q)										# bar_H_q					: (batch_size * max_article_sentence, comatch_bilstm_hidden_size, max_article_sentence_token)
		bar_H_a = torch.bmm(H_a_T.repeat(1, P.size(1), 1).view(-1, H_a_T.size(1), H_a_T.size(2)), G_a)										# bar_H_a					: (batch_size * N_CHOICES * max_article_sentence, comatch_bilstm_hidden_size, max_article_sentence_token)
		bar_H_q_T = bar_H_q.permute(0, 2, 1)																								# bar_H_q_T					: (batch_size * max_article_sentence, max_article_sentence_token, comatch_bilstm_hidden_size)
		bar_H_a_T = bar_H_a.permute(0, 2, 1)																								# bar_H_a_T					: (batch_size * N_CHOICES * max_article_sentence, max_article_sentence_token, comatch_bilstm_hidden_size)
		M_q = F.relu(self.W_m(torch.cat([bar_H_q_T - H_p, bar_H_q_T * H_p], axis=-1)))														# M_q						: (batch_size * max_article_sentence, max_article_sentence_token, comatch_bilstm_hidden_size)
		M_a = F.relu(self.W_m(torch.cat([bar_H_a_T - H_p_repeat_as_a, bar_H_a_T * H_p_repeat_as_a], axis=-1)))								# M_a						: (batch_size * N_CHOICES * max_article_sentence, max_article_sentence_token, comatch_bilstm_hidden_size)
		M_q_repeat_as_a = M_q.view(-1, P.size(1), M_q.size(1), M_q.size(2)).repeat(1, A.size(1), 1, 1).view(-1, M_q.size(1), M_q.size(2))	# M_q_repeat_as_a			: (batch_size * N_CHOICES * max_article_sentence, max_article_sentence_token, comatch_bilstm_hidden_size)
		C = torch.cat([M_q_repeat_as_a, M_a], axis=-1)																						# C							: (batch_size * N_CHOICES * max_article_sentence, max_article_sentence_token, 2 * comatch_bilstm_hidden_size)
		h_s_unpooled, _ = self.Encoder_C(C)																									# h_s_unpooled				: (batch_size * N_CHOICES * max_article_sentence, max_article_sentence_token, comatch_bilstm_hidden_size)
		h_s_unsqueezed = F.max_pool1d(h_s_unpooled.permute(0, 2, 1), kernel_size=self.p)													# h_s_unsqueezed			: (batch_size * N_CHOICES * max_article_sentence, comatch_bilstm_hidden_size, 1)
		h_s = h_s_unsqueezed.squeeze(-1)																									# h_s						: (batch_size * N_CHOICES * max_article_sentence, comatch_bilstm_hidden_size)
		H_s = h_s.view(-1, P.size(1), self.l)																								# H_s						: (batch_size * N_CHOICES,  max_article_sentence, comatch_bilstm_hidden_size)
		h_t_unpooled, _ = self.Encoder_H_s(H_s)																								# h_t_unpooled				: (batch_size * N_CHOICES, max_article_sentence, comatch_bilstm_hidden_size)
		h_t_unpooled_decomposed = h_t_unpooled.view(-1, A.size(1), P.size(1), self.l)														# h_t_unpooled_decomposed	: (batch_size, N_CHOICES, max_article_sentence, comatch_bilstm_hidden_size)
		
		# we do not use `max_pool1d` as the input of `max_pool1d` should be dim=2 or dim=3, while here is dim=4.
		h_t = torch.max(h_t_unpooled_decomposed.permute(0, 1, 3, 2), axis=-1)[0]															# h_t			: (batch_size, N_CHOICES, comatch_bilstm_hidden_size) 
		L_unactived = self.w(h_t).squeeze(-1)																								# L_unactived	: (batch_size, N_CHOICES)
		L = F.log_softmax(L_unactived, dim=-1)																								# L				: (batch_size, N_CHOICES)
		return L
