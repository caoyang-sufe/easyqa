# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn


a = dict(a=1,b=2,c=3)

for i, (k, v) in enumerate(a.items()):
	print(i, k, v)