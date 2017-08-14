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
	w1=np.random.normal(0,1/math.sqrt(input_dim),(input_dim,h))
	w2=np.random.normal(0,1/math.sqrt(h),(h,output_dim))
	b1=np.random.randn(1,h)
	b2=np.random.randn(1,output_dim)
	return w1,w2,b1,b2

def sigmoid(x):
	return 1/(1+np.exp(-x))

def div_sigmoid(x):
	return sigmoid(x)*(1-sigmoid(x))

def label_trans(train_label,output_dim):
	label=np.zeros((train_label.shape[0],output_dim))
	for i in range(label.shape[0]):
		label[i,train_label[i]]=1
	return label

def inference(X,w1,w2,b1,b2):
	''' forward progation

		inputs: 
		X: the input data
		returns:
		z: output of the network
	'''
	net1=np.dot(X,w1)+b1
	y=sigmoid(net1)
	net2=np.dot(y,w2)+b2
	z=sigmoid(net2)

	return y,z,net1,net2

def back_propagation(X,label,w1,w2,b1,b2,eta):
	y,z,net1,net2=inference(X,w1,w2,b1,b2)
	deta1=(z-label)*div_sigmoid(net2)
	delta_w2=np.dot(y.T,deta1)

	w2=w2-delta_w2*eta
	b2=b2-deta1*eta

	y,z,net1,net2=inference(X,w1,w2,b1,b2)
	deta1=(z-label)*div_sigmoid(net2)
	deta2=np.dot(deta1,w2.T)*div_sigmoid(net1)
	delta_w1=np.dot(X.T,deta2)

	return deta1,delta_w2,deta2,delta_w1




def online_train(train_data,train_label,w1,w2,b1,b2,eta):
	deta1=np.zeros((train_data.shape[0],b2.shape[1]))
	deta2=np.zeros((train_data.shape[0],b1.shape[1]))
	delta_w1=np.zeros((w1.shape[0],w1.shape[1],train_data.shape[0]))
	delta_w2=np.zeros((w2.shape[0],w2.shape[1],train_data.shape[0]))

	for i in range(train_data.shape[0]):
		deta1[i,:],delta_w2[:,:,i],deta2[i,:],delta_w1[:,:,i]=back_propagation(train_data[i,:].reshape((1,-1)),train_label[i,:].reshape((1,-1)),w1,w2,b1,b2,eta)

	deta1=deta1.sum(axis=0)
	deta2=deta2.sum(axis=0)
	delta_w1=delta_w1.sum(axis=2)
	delta_w2=delta_w2.sum(axis=2)

	w2=w1-delta_w2*eta
	b2=b2-deta1*eta
	w1=w1-delta_w1*eta
	b1=b1-deta2*eta

	return w1,w2,b1,b2

def trian(train_data,train_label,input_dim,h,output_dim,eta,trian_model):
	train_label=label_trans(train_label,output_dim)
	w1,w2,b1,b2=weigth_init(input_dim,h,output_dim)
	if trian_model=='online':
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
	elif trian_model=='batch':
		epoch=5
		for i in range(epoch):
			w1,w2,b1,b2=online_train(train_data,train_label,w1,w2,b1,b2,eta)
			print("the training epoch:%d"%(i+1))
		return w1,w2,b1,b2
	else :
		pass

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
		eta: float
			learn rate
		theta :float,a very small number
			the stop criteria
		train_model: ['batch','mini_batch','online']

		Atributte:
		----------
		w1,w2,b1,b2
	"""
	def __init__(self, input_dim,h,output_dim,eta,trian_model):
		self.input_dim=input_dim
		self.output_dim=output_dim
		self.h=h
		self.eta=eta
		self.trian_model=trian_model

	def fit(self,train_data,train_label):
		''' train the network

			Input:
			train :matrix ,[training_data,features],the training data
			train_label :vector,[label_1,....label_2],the label of the traing data

			return: self
		'''
		self.w1,self.w2,self.b1,self.b2=trian(train_data,train_label,self.input_dim,self.h,self.output_dim,self.eta,self.trian_model)

	def predict(self,test_data,test_label):
		'''calulate the accuracy on test data
			
			Inputs:
			test: matrix
			test_label :vector

			Outputs:correct_rate
		'''
		y,z,net1,net2=inference(test_data,self.w1,self.w2,self.b1,self.b2)
		predict_label=np.argmax(z,axis=1)
		flag=(predict_label==test_label)
		return flag.sum()/test_label.size


		


                

        
