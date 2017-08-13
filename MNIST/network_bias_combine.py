'''
this is the code for recgnition of MNIST with DNN

reference:Neural Network and Deep Learning

2017.8.11
'''
from __future__ import division
import numpy as np
import math
import random
######################


def weigth_init(input_dim,h,output_dim):
	w1=np.random.normal(0,1/math.sqrt(input_dim+1),(input_dim+1,h+1))
	w2=np.random.normal(0,1/math.sqrt(h+1),(h+1,output_dim))
	return w1,w2

def sigmoid(x):
	return 1/(1+np.exp(-x))

def div_sigmoid(x):
	return sigmoid(x)*(1-sigmoid(x))

def input_data_trans(input_data):
	return np.concatenate((input_data,np.ones((input_data.shape[0],1))),axis=1)

def label_trans(train_label,output_dim):
	label=np.zeros((train_label.shape[0],output_dim))
	for i in range(label.shape[0]):
		label[i,train_label[i]]=1
	return label

def inference(X,h,w1,w2):
	''' forward progation

		inputs: 
		X: the input data
		returns:
		z: output of the network
	'''
	input_data=input_data_trans(X)
	net1=np.dot(input_data,w1)
	y=sigmoid(net1)
	#hidden_input=np.append(y,1,axis=None).reshape((1,-1))
	net2=np.dot(y,w2)
	z=sigmoid(net2)

	return y,z,net1,net2

def back_propagation(X,label,h,w1,w2,learn_rate):
	y,z,net1,net2=inference(X,h,w1,w2)
	deta1=(z-label)*div_sigmoid(net2)
	delta_w2=np.dot(y.T,deta1)
	w2=w2-delta_w2*learn_rate

	y,z,net1,net2=inference(X,h,w1,w2)
	deta1=(z-label)*div_sigmoid(net2)
	deta2=np.dot(deta1,w2.T)*div_sigmoid(net1)
	input_data=input_data_trans(X)
	delta_w1=np.dot(input_data.T,deta2)
	w1=w1-delta_w1*learn_rate

	return w1,w2

def caculate_erro(X,h,w1,w2,w1_old,w2_old):
	input_data=input_data_trans(X)
	y,z,net1,net2=inference(X,h,w1,w2)
	y,z_old,net1,net2=inference(X,h,w1,w2)
	return math.sqrt(((z-z_old)**2).sum())


def trian(train_data,train_label,input_dim,h,output_dim,learn_rate,theta,trian_model):
	train_label=label_trans(train_label,output_dim)
	w1,w2=weigth_init(input_dim,h,output_dim)
	if trian_model=='random':
		error,count=(10,1)
		while error>=theta:
			index=random.randint(0,train_data.shape[0]-1)
			w1_old=w1
			w2_old=w2
			w1,w2=back_propagation(train_data[index,:].reshape((1,-1)),train_label[index,:].reshape((1,-1)),h,w1_old,w2_old,learn_rate)
			erro=caculate_erro(train_data[index,:].reshape((1,-1)),h,w1,w2,w1_old,w2_old)
			if count%5000==0:
				print("training times:%d"%count)
			count=count+1
		return w1,w2
	else:
		count=1
		while count<=5*train_data.shape[0]:
			index=random.randint(0,train_data.shape[0]-1)
			w1,w2=back_propagation(train_data[index,:].reshape((1,-1)),train_label[index,:].reshape((1,-1)),h,w1,w2,learn_rate)
			if count%5000==0:
				print("training times:%d"%count)
			count=count+1
		return w1,w2


class Network(object):
	"""the easiest version of deep Neural network

		Parameters:
		------------
		h: int
			the number of the hidden layer neurals
		input_dim : int
			the demension of input data
		output_dim : int
			the dimension of out put
		learn_rate: float
		theta :float,a very small number
			the stop criteria
		train_model

		Atributte:
		----------
		w1,w2
	"""
	def __init__(self, input_dim,h,output_dim,learn_rate,theta,trian_model):
		self.input_dim=input_dim
		self.output_dim=output_dim
		self.h=h
		self.learn_rate=learn_rate
		self.theta=theta
		self.trian_model=trian_model

	def fit(self,train_data,train_label):
		''' train the network

			Input:
			train :matrix ,[training_data,features],the training data
			train_label :vector,[label_1,....label_2],the label of the traing data

			return: self
		'''
		self.w1,self.w2=trian(train_data,train_label,self.input_dim,self.h,self.output_dim,self.learn_rate,self.theta,self.trian_model)

	def predict(self,test_data,test_label):
		'''calulate the accuracy on test data
			
			Inputs:
			test: matrix
			test_label :vector

			Outputs:correct_rate
		'''
		y,z,net1,net2=inference(test_data,self.h,self.w1,self.w2)
		predict_label=np.argmax(z,axis=1)
		flag=(predict_label==test_label)
		return flag.sum()/test_label.size


		


                

        
