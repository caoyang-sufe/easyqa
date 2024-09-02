# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

def f():
	# global question, context, answer
	question = "..."
	context = "..."
	answer = "..."
	columns = ["question", "context", "answer"]
	dic = {c: eval(c) for c in columns}
	print(dic)
	
def g():
	# global question, context, answer
	question = "..."
	context = "..."
	answer = "..."
	columns = ["question", "context", "answer"]
	dic = {}
	for c in columns:
		dic[c] = eval(c)
	print(dic)
	
g()