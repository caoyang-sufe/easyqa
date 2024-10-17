# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn
# Testscripts for src.module.comatch_module

import torch
from torch.nn import functional as F

from src.modules import CoMatch
from src.tests.modules.easy import summary

class CoMatchTest(CoMatch):
	"""Test if CoMatch is equivalent VerbosedCoMatch + CoMatchBranch"""
	def __init__(self, args):
		# Notice that the variables in CoMatch match those in VerbosedCoMatch and CoMatchBranch
		super(CoMatchTest, self).__init__(args=args)

	# Implementation of src.module.comatch_module.CoMatch
	def comatch(self, P, Q, A, P_shape, Q_shape, A_shape):
		return self.forward(P, Q, A, P_shape, Q_shape, A_shape)

	# Implementation of src.module.comatch_module.VerbosedCoMatch
	# Copy the code in forward function of rc.module.comatch_module.VerbosedCoMatch down here
	def verbosed_comatch(self, P, Q, A, P_shape, Q_shape, A_shape):
		L_unactived = list()
		for i in range(A.size(1)):
			H_s = torch.cat([self.comatch_branch(P[:, j, :].squeeze(1), Q,
												 A[:, i, :].squeeze(1),
												 P_shape[:, j], Q_shape,
												 A_shape[:, i]).unsqueeze(1) for j in range(P.size(1))], axis=1)		# H_s				: (batch_size, max_article_sentence, comatch_bilstm_hidden_size)
			h_t_i_unpooled, _ = self.Encoder_H_s(H_s)																	# h_t_i_unpooled	: (batch_size, max_article_sentence, comatch_bilstm_hidden_size)
			h_t_i_unsqueezed = F.max_pool1d(h_t_i_unpooled.permute(0, 2, 1), kernel_size=self.N)						# h_t_i_unsqueezed	: (batch_size, comatch_bilstm_hidden_size, 1)
			h_t_i = h_t_i_unsqueezed.squeeze(-1)																		# h_t_i				: (batch_size, comatch_bilstm_hidden_size)
			L_unactived.append(self.w(h_t_i))																			# self.w(h_t_i)		: (batch_size, 1)
		L_unactived = torch.cat(L_unactived, axis=-1)																	# L_unactived		: (batch_size, N_CHOICES)
		L = F.log_softmax(L_unactived, dim=-1)																			# L					: (batch_size, N_CHOICES)
		return L

	# Implementation of src.module.comatch_module.CoMatchBranch
	# Copy the code in forward function of rc.module.comatch_module.CoMatchBranch down here
	def comatch_branch(self, P, Q, A, P_shape, Q_shape, A_shape):
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

def testscript(epsilon=1e-6):
	from configs import ModuleConfig
	from src.tools.easy import load_args, update_args
	N_CHOICES = 4
	args = load_args(Config=ModuleConfig)
	kwargs = {"train_batch_size"			: 8,
			  "max_article_sentence"		: 4,
			  "max_article_sentence_token"	: 32,
			  "max_question_token"			: 16,
			  "max_option_token"			: 24,
			  "comatch_bilstm_hidden_size"	: 64,
			  "comatch_bilstm_num_layers"	: 2,
			  "comatch_bilstm_dropout"		: .3,
			  }
	update_args(args, **kwargs)
	# Generate a small input for quick test
	test_input = {'P'		: torch.rand(args.train_batch_size, args.max_article_sentence, args.max_article_sentence_token, args.comatch_embedding_size),
				  'Q'		: torch.rand(args.train_batch_size, args.max_question_token, args.comatch_embedding_size),
				  'A'		: torch.rand(args.train_batch_size, N_CHOICES, args.max_option_token, args.comatch_embedding_size),
				  "P_shape"	: torch.ones(args.train_batch_size, args.max_article_sentence).long() * 4,
				  "Q_shape"	: torch.ones(args.train_batch_size, ).long() * 4,
				  "A_shape"	: torch.ones(args.train_batch_size, N_CHOICES).long() * 4,
				  }
	comatch_test = CoMatchTest(args=args)
	summary(comatch_test)
	comatch_test.eval()
	comatch_output = comatch_test.comatch(**test_input)
	verbosed_comatch_output = comatch_test.verbosed_comatch(**test_input)
	error = torch.norm(comatch_output - verbosed_comatch_output, p="fro")
	print(f"""======== Co-Matching Test Report ========
Output of CoMatch:
{comatch_output}
Output of VerbosedCoMatch:
{verbosed_comatch_output}
Error:\t{error}
Result:\t{"Success" if error < epsilon else "Failure"}
=========================================""")
	return error < epsilon