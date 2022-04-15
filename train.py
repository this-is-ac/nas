import numpy as np
import csv

import tensorflow as tf
import tensorflow.contrib.slim as slim
from keras import backend as K
from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

from controller import Controller, StateSpace
from manager import NetworkManager
from model import model_fn

import scipy.signal
import random
import os
import sys
import pickle

import pandas as pd

from sklearn import svm
import datetime

policy_sess = tf.Session()
K.set_session(policy_sess)

NUM_LAYERS = 2  # number of layers of the state space
MAX_TRIALS = 50  # maximum number of models generated

MAX_EPOCHS = 1  # maximum number of epochs to train
CHILD_BATCHSIZE = 128  # batchsize of the child models
EXPLORATION = 0.8  # high exploration for the first 1000 steps
REGULARIZATION = 1e-3  # regularization strength
CONTROLLER_CELLS = 32  # number of cells in RNN controller
EMBEDDING_DIM = 20  # dimension of the embeddings for each state
ACCURACY_BETA = 0.8  # beta value for the moving average of the accuracy
CLIP_REWARDS = 0.0  # clip rewards in the [-0.05, 0.05] range
RESTORE_CONTROLLER = True  # restore controller to continue training

####################################################################################################

def unity(x) :
	return x

def negation(x) :
	return -x

def absolute(x) :
	return np.abs(x)

def power(x,k) :
	return np.power(x,k)

def sin(x) :
	return np.sin(x)

def cos(x) :
	return np.cos(x)

def tan(x) :
	return np.tan(x)

def exp(x) :
	return np.exp(x)

def log(x) :
	return np.log(x)

def sqrt(x) :
	return np.sqrt(np.abs(x))

def sinh(x) :
	return np.sinh(x)

def cosh(x) :
	return np.cosh(x)

def tanh(x) :
	return np.tanh(x)

def inverse_sinh(x) :
	return np.arcsinh(x)

def inverse_tan(x) :
	return np.arctan(x)             

def sinc(x) :
	return np.sinc(x)

def sigmoid(x) :
	return 1.0/(1.0 + np.exp(-x))

def maximum(x,y) :
	return np.maximum(x,y)

def minimum(x,y) :
	return np.minimum(x,y)

def erf(x) :
	return scipy.special.erf(x)

def summation(x,y) :
	return np.add(x,y)

def difference(x,y) :
	return np.subtract(x,y)

def multiply(x,y) :
	return np.multiply(x,y)

def divide(x,y) :
	return np.divide(x,y)           

def norm(x, k) :
	if np.array(x).shape[-1] == 1 :
		return x
	return np.linalg.norm(x, ord = k, axis = -1, keepdims = True)*gamma(x, 1)

def gamma(x, y) :
	return 1.0/float(train_data.shape[1])

def squared_diff(x, y) :
	return absolute(difference(x**2, y**2))

def squared_sum(x, y) :
	return summation(x**2, y**2)

def dot_prod(x, y) :
	if np.array(x).shape[-1] == 1 and np.array(y).shape[-1] == 1 :
		return multiply(x, y)
	return np.sum(multiply(x, y), axis = -1, keepdims = True)*gamma(x, y)

def norm1(x, k) :
	if np.array(x).shape[-1] == 1 :
		return x
	return np.linalg.norm(x, ord = k, axis = -1)

####################################################################################################

state_space = StateSpace()

state_space.add_state(name='operator1', values=[lambda x, y : absolute(difference(x, y)), summation, squared_diff, lambda x, y : norm(difference(x, y), 1), lambda x, y : norm(difference(x, y), 2), dot_prod, multiply])
state_space.add_state(name='operator2', values=[lambda x, y : absolute(difference(x, y)), summation, squared_diff, lambda x, y : norm(difference(x, y), 1), lambda x, y : norm(difference(x, y), 2), dot_prod, multiply])
state_space.add_state(name='unaryOp1', values=[unity, negation, absolute, lambda x : power(x,2), lambda x : power(x,3), sqrt, exp, sin, cos, sinh, cosh, tanh, lambda x : maximum(x,0), lambda x : minimum(x,0), sigmoid, lambda x : log(1 + exp(x)), lambda x : norm(x, 1), lambda x : norm(x, 2)])
state_space.add_state(name='unaryOp2', values=[unity, negation, absolute, lambda x : power(x,2), lambda x : power(x,3), sqrt, exp, sin, cos, sinh, cosh, tanh, lambda x : maximum(x,0), lambda x : minimum(x,0), sigmoid, lambda x : log(1 + exp(x)), lambda x : norm(x, 1), lambda x : norm(x, 2)])
state_space.add_state(name='binaryOp', values=[summation, multiply, difference, maximum, minimum,dot_prod, lambda x,y : exp(-1 * absolute(difference(x,y))), lambda x,y : x])

state_space.print_state_space()

df = pd.read_csv('Merged.csv', encoding= 'unicode_escape')
df = df[:-2]

from sklearn.preprocessing import MinMaxScaler

split = 0.8

scaler_x2 = MinMaxScaler()
scaler_x3 = MinMaxScaler()
scaler_x4 = MinMaxScaler()

x2 = df["Air Temperature"].to_numpy()
x2 = np.reshape(x2,(-1,1))

x3 = df["Global Horizontal"].to_numpy()
x3 = np.reshape(x3,(-1,1))

x4 = df["Diffuse Horizontal"].to_numpy()
x4 = np.reshape(x4,(-1,1))

n = len(x2)

x2_train = x2[:int(split * n), :]
x2_test = x2[int(split * n):, :]

x3_train = x3[:int(split * n), :]
x3_test = x3[int(split * n):, :]

x4_train = x4[:int(split * n), :]
x4_test = x4[int(split * n):, :]

scaler_x2.fit(x2_train)
x2_train = scaler_x2.transform(x2_train)
x2_test = scaler_x2.transform(x2_test)

scaler_x3.fit(x3_train)
x3_train = scaler_x3.transform(x3_train)
x3_test = scaler_x3.transform(x3_test)

scaler_x4.fit(x4_train)
x4_train = scaler_x4.transform(x4_train)
x4_test = scaler_x4.transform(x4_test)

scaler_y = MinMaxScaler()
Y = df['Direct Normal'].to_numpy()
Y = np.reshape(Y,(-1,1))
n = len(Y)
Y_train = Y[:int(split * n)]
Y_test = Y[int(split * n):]
scaler_y.fit(Y_train)
Y_train = scaler_y.transform(Y_train)
Y_test = scaler_y.transform(Y_test)

X_train = np.concatenate((x2_train, x3_train, x4_train), axis = 1)
X_test = np.concatenate((x2_test, x3_test, x4_test), axis = 1)

dataset = [X_train, Y_train, X_test, Y_test]

previous_acc = 0.0
total_reward = 0.0

with policy_sess.as_default():
    controller = Controller(policy_sess, NUM_LAYERS, state_space,
                            reg_param=REGULARIZATION,
                            exploration=EXPLORATION,
                            controller_cells=CONTROLLER_CELLS,
                            embedding_dim=EMBEDDING_DIM,
                            restore_controller=RESTORE_CONTROLLER)

manager = NetworkManager(dataset, epochs=MAX_EPOCHS, child_batchsize=CHILD_BATCHSIZE, clip_rewards=CLIP_REWARDS,
                         acc_beta=ACCURACY_BETA)

state = state_space.get_random_state_space(NUM_LAYERS)
print("Initial Random State : ", state_space.parse_state_space_list(state))
print()

controller.remove_files()

for trial in range(MAX_TRIALS):
    with policy_sess.as_default():
        K.set_session(policy_sess)
        actions = controller.get_action(state)  # get an action for the previous state

    # print the action probabilities
    state_space.print_actions(actions)
    print("Predicted actions : ", state_space.parse_state_space_list(actions))

    # build a model, train and get reward and accuracy from the network manager
    reward, previous_acc = manager.get_rewards(model_fn, state_space.parse_state_space_list(actions))
    print("Rewards : ", reward, "Accuracy : ", previous_acc)

    with policy_sess.as_default():
        K.set_session(policy_sess)

        total_reward += reward
        print("Total reward : ", total_reward)

        # actions and states are equivalent, save the state and reward
        state = actions
        controller.store_rollout(state, reward)

        # train the controller on the saved state and the discounted rewards
        loss = controller.train_step()
        print("Trial %d: Controller loss : %0.6f" % (trial + 1, loss))

        # write the results of this trial into a file
        # loss - previous_accuracy - reward - actions/state
        with open('train_history.csv', mode='a+') as f:
            data = [loss, previous_acc, reward]
            data.extend(state_space.parse_state_space_list(state))
            writer = csv.writer(f)
            writer.writerow(data)
    print()

print("Total Reward : ", total_reward)