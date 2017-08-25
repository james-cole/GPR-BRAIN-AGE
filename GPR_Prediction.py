from __future__ import print_function
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


class Gaussian_Process_Regression(object):

	def __init__(self,num_test,dim_input,dim_output,age_mean,num_data):
       
		self.Y_train = tf.placeholder(tf.float64,shape=(num_data,dim_output))
		self.X_train = tf.placeholder(tf.float64,shape=(num_data,dim_input))
		self.X_test = tf.placeholder(tf.float64,shape=(num_test,dim_input))
		self.Y_test = tf.placeholder(tf.float64,shape=(num_test,dim_output))
		self.age_mean = age_mean

	def restore_model(self):      
		self.restored_sess = tf.Session()
		new_saver = tf.train.import_meta_graph('./logs/GPR_model.meta')
		new_saver.restore(self.restored_sess,tf.train.latest_checkpoint('./logs/'))  
		self.variance_output = tf.get_collection('variance_output')[0]
		self.variance_kernel = tf.get_collection('variance_kernel')[0]
		print('we print variance_output')
		print(self.restored_sess.run(self.variance_output))
		print('we print variance kernel')
		print(self.restored_sess.run(self.variance_kernel))
		self.Kuu= tf.get_collection('sim_matrix')[0]
		print('we restored the variables from the previous model')

	def chol_solve(self,L,X):

		return tf.matrix_triangular_solve(tf.transpose(L),tf.matrix_triangular_solve(L,X),lower=False)

	def chol_solve_reverse(self,L,X):
    
		return tf.transpose(tf.matrix_triangular_solve(tf.transpose(L),tf.matrix_triangular_solve(L,tf.transpose(X)),lower=False))

	def eye(self,N):
        
		return tf.diag(tf.ones(tf.stack([N,]),dtype=tf.float64))

	def condition(self,X):

		return X + self.eye(tf.shape(X)[0]) * 1e-6


	def RBF(self,X1,X2):

                return tf.matmul(X1 * tf.exp(self.variance_kernel),X2,transpose_b=True)



	def build_predict(self,X_new,full_cov=False):
    
		print('lets print Kuu')
		print(str(self.restored_sess.run(self.Kuu)))
		t2=time.time()
		Kx = self.RBF(self.X_train, X_new)
		#Kuu = self.RBF(self.X_train,self.X_train)
		L = tf.cholesky(self.condition(self.Kuu))
		A = tf.matrix_triangular_solve(L, Kx, lower=True)
		V = tf.matrix_triangular_solve(L, self.Y_train)

		fmean = tf.matmul(A, V, transpose_a=True) + self.age_mean
		'''
		if full_cov:
			fvar = self.RBF(X_new,X_new) - tf.matmul(A, A, transpose_a=True) + self.variance_output * self.eye(X_new.shape[0])
		else:
			fvar = tf.diag(tf.diag_part(self.RBF(X_new,X_new) - tf.matmul(A, A, transpose_a=True))+self.variance_output*self.eye(X_new.shape[0]) )
		'''
		t3=time.time()
		timer(t2,t3)
		
		return fmean

	def session_TF(self,X_testing,Y_testing,X_training,Y_training):

		self.restore_model()
		predictions = self.build_predict(self.X_test)		
		predictions_now = self.restored_sess.run(predictions,feed_dict={self.X_test:X_testing,self.X_train:X_training,self.X_train:X_training,self.Y_train:Y_training})
		rmse_now = np.sqrt(mse(predictions_now,Y_testing))
		print('rmse at test time is:'+str(rmse_now))
		mae_now = mae(predictions_now,Y_testing)
		print('mae at test time is:'+str(mae_now))
		
		Kuu_val,Kuu_val_reloaded = self.restored_sess.run([self.RBF(self.X_train,self.X_train),self.Kuu],feed_dict={self.X_train:X_training})
		mata = Kuu_val - Kuu_val_reloaded
		print(np.mean(mata))


if __name__=='__main__':


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

        lista_initializare = [1.0 for n in range(X_training.shape[1])]
        lista_initializare = np.asarray(lista_initializare)
        t3 = time.time()
        print('this is how much it takes to initialize the legnthscales')
        timer(t2,t3)


	obiect =  Gaussian_Process_Regression(num_test=X_testing.shape[0],dim_input=X_testing.shape[1],dim_output=Y_testing.shape[1],age_mean=age_mean,num_data=X_training.shape[0])
	obiect.session_TF(X_testing=X_testing,Y_testing=Y_testing,X_training=X_training,Y_training=Y_training)
	#print(perm[1900:])



	
