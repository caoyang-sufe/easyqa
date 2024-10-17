# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn

def summary(model,
			num_blank_of_param_name = 15,
			num_blank_of_param_size = 10,
			num_blank_of_param_num	= 2,
			param_name_string 		= "Param Name",
			param_size_string 		= "Param Size",
			param_num_string 		= "Param #"):
	blank = ' '
	num_dash = (num_blank_of_param_name + num_blank_of_param_size + num_blank_of_param_num) * 2 + len(param_name_string) + len(param_size_string) + len(param_num_string) + 4
	print('-' * num_dash)
	print('|' + blank * num_blank_of_param_name + param_name_string + blank * num_blank_of_param_name + "|" + blank * num_blank_of_param_size + param_size_string + blank * num_blank_of_param_size + "|" + blank * num_blank_of_param_num + param_num_string + blank * num_blank_of_param_num + "|")
	print('-' * num_dash)
	num_param = 0
	type_size = 1
	for name, parameter in model.named_parameters():
		if len(name) <= num_blank_of_param_name * 2 + len(param_name_string):
			num_blank_left = int((num_blank_of_param_name * 2 + len(param_name_string) - len(name)) / 2) - 2
			num_blank_right = num_blank_of_param_name * 2 + len(param_name_string) - len(name) - num_blank_left - 2
			name = num_blank_left * blank + name + num_blank_right * blank
		size = str(parameter.size())
		if len(size) <= num_blank_of_param_size * 2 + len(param_size_string):
			num_blank_left = int((num_blank_of_param_size * 2 + len(param_size_string) - len(size)) / 2) - 2
			num_blank_right = num_blank_of_param_size * 2 + len(param_size_string) - len(size) - num_blank_left- 2
			size = num_blank_left * blank + size + num_blank_right * blank
		num_param_each = 1
		for k in parameter.shape:
			num_param_each *= k
		num_param += num_param_each
		num_param_each = str(num_param_each)
		if len(num_param_each) <= num_blank_of_param_num * 2 + len(param_num_string):
			num_blank_left = int((num_blank_of_param_num * 2 + len(param_num_string) - len(num_param_each)) / 2) - 2
			num_blank_right = num_blank_of_param_num * 2 + len(param_num_string) - len(num_param_each) - num_blank_left - 2
			num_param_each = num_blank_left * blank + num_param_each + num_blank_right * blank
		print(f"| {name} | {size} | {num_param_each} |")
	print('-' * num_dash)
	print("The total number of parameters: " + str(num_param))
	print(f"The parameters of Model {model._get_name()}: {round(num_param * type_size / 1000 / 1000, 4)}M")
	print('-' * num_dash)
