# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

class A:

	def __init__(self):
		pass

	def f(self, a):

		return a ** 2

class B(A):

	def __init__(self):

		super(B, self).__init__()

	def f(self, a):

		return super(B, self).f(a) ** 2

a = A()

print(a.f(2))

b = B()

print(b.f(2))