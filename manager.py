import numpy as np

from sklearn import svm

from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class NetworkManager:
    def __init__(self, dataset, epochs=5, child_batchsize=128, acc_beta=0.8, clip_rewards=0.0):
        '''
            acc_beta: exponential weight for the accuracy
            clip_rewards: float - to clip rewards in [-range, range] to prevent large weight updates. Use when training is highly unstable.
        '''
        self.dataset = dataset
        self.epochs = epochs
        self.batchsize = child_batchsize
        self.clip_rewards = clip_rewards

        self.beta = acc_beta
        self.beta_bias = acc_beta
        self.moving_acc = 0.0

    def get_rewards(self, model_fn, actions, cust_train_data, NUM_UNITS):
        '''
        Returns a reward for training a model with the given actions
        '''
        with tf.Session(graph=tf.Graph()) as network_sess:
            K.set_session(network_sess)

            train_k_matrix, val_k_matrix = model_fn(actions, cust_train_data, NUM_UNITS)

            train_train_x, train_train_y, train_labels, validation_train_x, validation_train_y, val_labels = self.dataset

            clf = svm.SVR(kernel="precomputed")
            clf.fit(train_k_matrix, train_labels)
            predictions = clf.predict(val_k_matrix)

            loss = mean_squared_error(val_labels, predictions, squared=False)
            acc = r2_score(val_labels, predictions)

            reward = (acc - self.moving_acc)

            # if rewards are clipped, clip them in the range -0.05 to 0.05
            if self.clip_rewards:
                reward = np.clip(reward, -0.05, 0.05)

            # update moving accuracy with bias correction for 1st update
            if self.beta > 0.0 and self.beta < 1.0:
                self.moving_acc = self.beta * self.moving_acc + (1 - self.beta) * acc
                self.moving_acc = self.moving_acc / (1 - self.beta_bias)
                self.beta_bias = 0

                reward = np.clip(reward, -0.1, 0.1)

            print()
            print("Manager: EWA Accuracy = ", self.moving_acc)

        network_sess.close()

        return reward, acc