# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from collections import defaultdict
import sys
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import time

def timer(start,end):
       hours, rem = divmod(end-start, 3600)
       minutes, seconds = divmod(rem, 60)
       print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


class GaussianProcessRegression(object):

	def __init__(self,num_data,num_test,dim_input,dim_output,lista_initializare,age_mean,num_validation):

		self.num_data = num_data
		self.num_test = num_test
		self.dim_input = dim_input
		self.dim_output = dim_output
		self.age_mean = age_mean
		self.sess= tf.Session()
		self.X_train = tf.placeholder(tf.float64,shape=(num_data,dim_input))
		self.Y_train = tf.placeholder(tf.float64,shape=(num_data,dim_output))

		self.X_validation = tf.placeholder(tf.float64,shape=(num_validation,dim_input))
		self.Y_validation = tf.placeholder(tf.float64,shape=(num_validation,dim_output))		

		self.X_test = tf.placeholder(tf.float64,shape=(num_test,dim_input))
		self.Y_test = tf.placeholder(tf.float64,shape=(num_test,dim_output))
		self.variance_output = tf.Variable(0.0,dtype=tf.float64,name='likelihood_variance',trainable=True)
		tf.add_to_collection('variance_output',self.variance_output)
		self.variable_summaries(self.variance_output,'log_var_output')
		lista_initializare = tf.log(lista_initializare)            
		self.variance_kernel = tf.Variable(0.0,dtype=tf.float64)
		self.variable_summaries(self.variance_kernel,'log_variance_kernel')
		#self.lengthscales = tf.Variable(lista_initializare,dtype=tf.float64)
		

		#self.variable_summaries(self.lengthscales,'log_lenghtscales')
		tf.add_to_collection('variance_kernel',self.variance_kernel)
		#tf.add_to_collection('lengthscales',self.lengthscales)
		self.similarity_matrix = tf.Variable(tf.zeros(shape=(num_data,num_data),dtype=tf.float64),trainable=False)
		tf.add_to_collection('sim_matrix',self.similarity_matrix)
		self.K = tf.placeholder(tf.float64,shape=(num_data,num_data))
		self.assign_op = tf.assign(self.similarity_matrix,self.K)

		self.saver = tf.train.Saver(var_list=tf.get_collection('variance_kernel')+tf.get_collection('variance_output')+tf.get_collection('sim_matrix'))


	def variable_summaries(self,var,name):
		"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
		with tf.name_scope(name):
			mean = tf.reduce_mean(var)
			tf.summary.scalar('mean', mean)
			with tf.name_scope('stddev'):
				stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
			tf.summary.scalar('stddev', stddev)
			tf.summary.scalar('max', tf.reduce_max(var))
			tf.summary.scalar('min', tf.reduce_min(var))
			tf.summary.histogram('histogram', var)


	def chol_solve(self,L,X):

		return tf.matrix_triangular_solve(tf.transpose(L),tf.matrix_triangular_solve(L,X),lower=False)

	def chol_solve_reverse(self,L,X):
    
		return tf.transpose(tf.matrix_triangular_solve(tf.transpose(L),tf.matrix_triangular_solve(L,tf.transpose(X)),lower=False))    

	
	def eye(self,N):
        
		return tf.diag(tf.ones(tf.stack([N,]),dtype=tf.float64))


	def condition(self,X):

		return X + self.eye(tf.shape(X)[0]) * 1e-6

	def cholesky_solver(self,X,RHS):

		X_conditioned = self.condition(X)
		chol = tf.cholesky(X_conditioned)
		return tf.cholesky_solve(chol,RHS)

	def cholesky_solver_reverse(self,X,RHS):
        
		X_conditioned = self.condition(X)
		chol = tf.cholesky(X_conditioned)        
		return tf.transpose(tf.cholesky_solve(chol,tf.transpose(RHS)))

	def RBF(self,X1,X2):
            
		'''
		X1 = X1 / tf.exp(self.lengthscales)
		X2 = X2 / tf.exp(self.lengthscales)
		X1s = tf.reduce_sum(tf.square(X1),1)
		X2s = tf.reduce_sum(tf.square(X2),1)       

		return tf.exp(self.variance_kernel) * tf.exp(-(-2.0 * tf.matmul(X1,tf.transpose(X2)) + tf.reshape(X1s,(-1,1)) + tf.reshape(X2s,(1,-1)))/2)      
		'''
		return tf.matmul(X1 * tf.exp(self.variance_kernel),X2,transpose_b=True)



	def multivariate_normal(self,x,L):
    	
		alpha = self.chol_solve(L, x)
		num_col = tf.cast(tf.shape(x)[1],tf.float64)
    		
		num_dims = tf.cast(tf.shape(x)[0],tf.float64)
		ret = 0.5 * num_dims * num_col * np.log(2 * np.pi)
		ret += num_col * tf.reduce_sum(tf.log(tf.matrix_diag_part(L)))
		ret += 0.5 * tf.matmul(tf.transpose(self.Y_train),alpha)
		return ret

	def build_likelihood(self):
        	
		K = self.RBF(self.X_train,self.X_train) + tf.eye(tf.shape(self.X_train)[0], dtype=tf.float64) * tf.exp(self.variance_output) 
		L = tf.cholesky(self.condition(K))
        
		return self.multivariate_normal(self.Y_train,L)


	def build_predict(self,X_new,full_cov=False):
    
		Kx = self.RBF(self.X_train, X_new)
		K = self.RBF(self.X_train,self.X_train) + tf.eye(tf.shape(self.X_train)[0], dtype=tf.float64) * tf.exp(self.variance_output)
		L = tf.cholesky(self.condition(K))
		A = tf.matrix_triangular_solve(L, Kx, lower=True)
		V = tf.matrix_triangular_solve(L, self.Y_train)

		fmean = tf.matmul(A, V, transpose_a=True)
        
		if full_cov:
			fvar = self.RBF(X_new,X_new) - tf.matmul(A, A, transpose_a=True)
		else:
			fvar = tf.diag(tf.diag_part(self.RBF(X_new,X_new) - tf.matmul(A, A, transpose_a=True)))
	
		return fmean,fvar
	
	def predict_f_samples_full_cov(self, Xnew):
		"""
		Produce samples from the posterior latent function(s) at the points
		Xnew.
		"""
		mu,var = self.build_predict(Xnew, full_cov=True)
        
		X_new_number = tf.cast(tf.shape(Xnew)[0],tf.int32)
		L = tf.cholesky(self.condition(var))
        
		V = tf.random_normal(shape=(X_new_number,1),dtype=tf.float64)
		rezultat = mu + tf.matmul(L, V)

		return rezultat

	def session_TF(self,X_training,Y_training,X_testing,Y_testing,X_validation,Y_validation):

		cost = self.build_likelihood()
		tf.summary.scalar('cost',tf.squeeze(cost))
		opt =tf.train.AdamOptimizer(0.01)
		train_op = opt.minimize(cost)
		#predictions = self.predict_f_samples_full_cov(self.X_test)
		predictions = self.build_predict(self.X_test)
		predictions_training = self.build_predict(self.X_train)
		predictions_validation = self.build_predict(self.X_validation)
		rmse_training = tf.sqrt(tf.reduce_mean(tf.squared_difference(self.Y_train,predictions_training[0])))
		rmse_validation = tf.sqrt(tf.reduce_mean(tf.squared_difference(self.Y_validation,predictions_validation[0])))
		tf.summary.scalar('rmse_training',tf.squeeze(rmse_training))
		tf.summary.scalar('rmse_validation',tf.squeeze(rmse_validation))

		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter('./tensorboard/',self.sess.graph)
		self.sess.run(tf.global_variables_initializer())
		for i in range(1000):

			_,costul_actual,summary  = self.sess.run([train_op,cost,merged],feed_dict={self.X_train:X_training,self.Y_train:Y_training,self.X_validation:X_validation,self.Y_validation:Y_validation})
			train_writer.add_summary(summary,i)
			rmse_validation_now = self.sess.run(rmse_validation,feed_dict={self.X_validation:X_validation,self.Y_validation:Y_validation,self.X_train:X_training,self.Y_train:Y_training})
			rmse_training_now = self.sess.run(rmse_training,feed_dict={self.X_train:X_training,self.Y_train:Y_training})
			print('at iteration '+str(i) + ' we have nll : '+str(costul_actual) +' **** rmse training : ' + str(rmse_training_now) + ' **** rmse validation : '+str(rmse_validation_now))

		predictii,vars = self.sess.run(predictions,feed_dict={self.X_test:X_testing,self.X_train:X_training,self.Y_train:Y_training})
		#for i in range(self.num_test):
		#print 'real data : '+str(Y_testing[i]) + ' *** predicted : '+str(predictii[i] + self.age_mean)
		print('****RMSE at testing time*****')
		rmse_now = np.sqrt(mse(Y_testing,predictii+self.age_mean))
		print(rmse_now)
		print('****MAE at testing time*****')
		mae_now = mae(Y_testing,predictii+self.age_mean)
		print(mae_now)
		text_printat=''
		text_printat+='**RMSE at testing time is :'+str(rmse_now)+' \n'
		text_printat+='***MAE at testing time is :'+str(mae_now)+' \n'
		with open('./results/global_age','w') as file:
			file.write(text_printat)


		K_mata = self.sess.run(self.RBF(self.X_train,self.X_train) + tf.eye(tf.shape(self.X_train)[0], dtype=tf.float64) * tf.exp(self.variance_output),feed_dict={self.X_train:X_training})
		self.sess.run(self.assign_op,feed_dict={self.K:K_mata})

		self.saver.save(self.sess,'./logs/GPR_model')

		print('variance_kernel')
		print(self.sess.run(self.variance_kernel))
		print('variance_output')
		print(self.sess.run(self.variance_output))






if __name__ == '__main__':

	t0 = time.time()
	Y_total = np.genfromtxt('/home/sebict/brain_data/Y_original.txt',dtype=np.float64)
	Y_total = Y_total.reshape((2001,1))
	print('shape of Y')
	print(Y_total.shape)
	X_total = np.genfromtxt('/home/sebict/brain_data/X_original.txt',dtype=np.float64)
	print('shape of X')
	print(X_total.shape)
	np.random.seed(7)
	perm = np.random.permutation(X_total.shape[0])
	Y_total = Y_total[perm]
	X_total = X_total[perm]
	# prepare input data
	X_training = X_total[:1600,:]
	X_validation = X_total[1600:1800,:]
	X_testing = X_total[1800:,:]
	print('shape of X_training')
	print(X_training.shape)

	# prepare output data
	Y_training = Y_total[:1600,:]
	Y_validation = Y_total[1600:1800,:]
	Y_testing  =Y_total[1800:,:]
	print('shape of Y_training')
	print(Y_training.shape)	

	age_mean  = np.mean(Y_training)
	Y_training = Y_training - age_mean
	Y_validation  =Y_validation - age_mean
	t1 = time.time()
	print('this is how much it takes to load the data')
	timer(t0,t1)
	t2=time.time()
	'''
	lista_init_local= []
	for j in range(X_training.shape[1]):

		t = X_training[:,j].reshape((1600,))
		t = np.reshape(t,(len(t),1))
		matrice_diff = abs(t - t.transpose())
		lista_init_local.append(matrice_diff.mean())
	lista_init_local = np.asarray(lista_init_local)
	lista_initializare = lista_init_local.astype(np.float64)
	'''
	lista_initializare = [1.0 for n in range(X_training.shape[1])]
	lista_initializare = np.asarray(lista_initializare)
	t3 = time.time()
	print('this is how much it takes to initialize the legnthscales')
	timer(t2,t3)

	t4=time.time()
	obiect =  GaussianProcessRegression(num_data = X_training.shape[0],num_test=X_testing.shape[0],dim_input=X_training.shape[1],dim_output=Y_training.shape[1],lista_initializare=lista_initializare,age_mean=age_mean,num_validation=X_validation.shape[0])
	obiect.session_TF(X_training=X_training,X_testing=X_testing,Y_training=Y_training,Y_testing=Y_testing,X_validation=X_validation,Y_validation=Y_validation)
	t5=time.time()
	print('this is how much it took to train the model')
	timer(t4,t5)






