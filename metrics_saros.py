import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import sys
import numpy as np
import math
import time
import random
import subprocess
import shlex
from subprocess import call



def tf_cross_entropy(raw_margins, target_values,trunc_max=100):
    elementwise_entropy_loss = -tf.multiply(target_values,tf.log(raw_margins))-\
                                 tf.multiply(1-target_values,tf.log(1-raw_margins))
    checked_elwise_loss = tf.verify_tensor_all_finite(elementwise_entropy_loss, 
                                                      msg='NaN or Inf in loss vector', name='checked_elwise_ce')
    mean_loss = tf.reduce_mean(tf.minimum(checked_elwise_loss, trunc_max))
    return mean_loss


def tf_mean_l2(w):
    elementwise_sq_norm = tf.reduce_sum(tf.pow(w, 2), axis=1)
    checked_elwise_l2 = tf.verify_tensor_all_finite(elementwise_sq_norm, msg='NaN or Inf in norm', name='checked_elwise_l2')
    mean_l2 = tf.reduce_mean(checked_elwise_l2)
    return mean_l2


class SAROS(object):
    
    def __init__(self, n_users, n_items, n_embeddings, alpha_reg=0, seed=None):
        self.N_USERS = n_users
        self.N_ITEMS = n_items
        self.N_EMBEDDINGS = n_embeddings
        self.alpha_reg = alpha_reg
        self.seed = seed
        self.graph = tf.Graph()
        if seed:
            self.graph.seed = seed


    def build_graph(self):
        with self.graph.as_default():
            # placeholders
            self.user_ids = tf.placeholder(tf.int32, (None,), name='user_ids')
            self.left_ids = tf.placeholder(tf.int32, (None,), name='left_ids')
            self.right_ids = tf.placeholder(tf.int32, (None,), name='right_ids')
            self.target_y = tf.placeholder(tf.float32, (None,), name='target_y')
                                      
                                                                      
            # main parameters
            self.user_latents = tf.Variable(tf.random_uniform(shape=(self.N_USERS, self.N_EMBEDDINGS),seed=123),trainable=True, name='user_latents')
            self.item_latents = tf.Variable(tf.random_uniform(shape=(self.N_ITEMS, self.N_EMBEDDINGS),seed=124),trainable=True, name='item_latents')
                                      
                                      
            # get embeddings
            self.embedding_user = tf.nn.embedding_lookup(self.user_latents,self.user_ids, name='embedding_user')
            self.embedding_left = tf.nn.embedding_lookup(self.item_latents,self.left_ids, name='embedding_left')
            self.embedding_right = tf.nn.embedding_lookup(self.item_latents,self.right_ids,name='embedding_right')
            self.embedding_mul = tf.multiply(self.embedding_left,self.embedding_user)
                                      
            # raw margins for primal ranking loss
            self.embedding_diff = self.embedding_left - self.embedding_right
            self.relevances = tf.reduce_sum(tf.multiply(self.embedding_user, self.embedding_left), axis=1)
                                      
            self.embedding_margins = tf.reduce_sum(tf.multiply(self.embedding_user, self.embedding_diff),axis=1, name='embedding_margins')
            self.embedding_loss = tf_cross_entropy(tf.sigmoid(self.embedding_margins),(self.target_y+1.)/2)
                                      
            # outs
            self.regularization = tf_mean_l2(self.embedding_user) + tf_mean_l2(self.embedding_left) + tf_mean_l2(self.embedding_right)
            self.target = self.embedding_loss + self.alpha_reg * self.regularization
            
            self.opt = tf.train.GradientDescentOptimizer(learning_rate = 0.005)
            self.train = self.opt.minimize(self.target)   
                                                                                                                                                                                                         
            self.init_all_vars = tf.global_variables_initializer()
                                                                                                                                                

    @property
    def weights_i(self):
        return self.user_latents.eval(session=self.session)
    
    @property
    def weights_u(self):
        return self.item_latents.eval(session=self.session)

    
    def initialize_session(self):
        config = tf.ConfigProto()
        # for reduce memory allocation
        config.gpu_options.allow_growth = True
        self.session = tf.Session(graph=self.graph, config=config)
        self.session.run(self.init_all_vars)
    
    def destroy(self):
        self.session.close()
        self.graph = None



latent_dim = 4 #embeddings size
train_part = 0.8

#saving all negatives and positives for each user
def get_negatives_and_positives_from_user(it_for_u, clicks):
    neg_all_us = []
    pos_all_us = []
    
    for n in range(len(clicks)):
        
        if (clicks[n] == 0):
            neg_all_us = neg_all_us + [it_for_u[n]]
        
        if (clicks[n] == 1):
            pos_all_us = pos_all_us + [it_for_u[n]]
            
    return pos_all_us, neg_all_us

#build triplets for test predictions
def build_all_triplets_for_loss(positives, negatives):
    neg_triplets = negatives*len(positives)
    pos_triplets = []
    for it in positives:
        pos_triplets += [it]*len(negatives)
    return pos_triplets, neg_triplets

# Read data
df = pd.read_csv('/home/sburashnikova/SAROS/datasets/ml_100_data',sep=',',header=None)
#Extract the items for each user and create the list of clicked and non-clicked items for each user
users = set(df[0])
users = list(users)
items_for_user = []
neg_all_test = []
pos_all_test = []
us_list_test = []
num_items = 0
clicks_all = []
neg_items_train = []

for user in users:
    df_user  = df[df[0]==user].sort_values(by=[3])#subdataset for each user, sorted by timestamp
    click = df_user[2]
    click = list(click)
    it_for_u = df_user[1]
    it_for_u = list(it_for_u)
    num_items = max(max(it_for_u), num_items)
    items_for_user.append(it_for_u)# list of items for each user
    clicks = []    
    for j in click:        
        if (j >= 4):
            clicks = clicks + [1]
        else:
            clicks = clicks + [0]
    
    clicks_all.append(clicks)
    
    #getting all positives and negatives items for each user for train and test
    train_ind = int(train_part*len(clicks))
    pos_us_train, neg_us_train = get_negatives_and_positives_from_user(it_for_u[:train_ind], clicks[:train_ind])
    neg_items_train.append(neg_us_train)
    pos_us_test, neg_us_test= get_negatives_and_positives_from_user(it_for_u[train_ind:], clicks[train_ind:])
    
    #getting triplets for test
    pos_triplets, neg_triplets = build_all_triplets_for_loss(pos_us_test, neg_us_test)
    pos_all_test += pos_triplets
    neg_all_test += neg_triplets
    us_list_test += [user]*len(pos_triplets)


#creating the files for saving results
export_basename = '/home/sburashnikova/SAROS/results/'

#Initialize the model
num_users_all = max(users)+1
num_items += 1

model = SAROS(num_users_all, num_items, latent_dim, alpha_reg=0.01)
model.build_graph()
model.initialize_session()

####################### Build triplets for training #####################

all_users_triplets = []
for i in range(len(users)):

    train_ind = int(train_part * len(items_for_user[i]))
    test_ind = np.arange(train_ind, len(items_for_user[i]))
    
    #skip user if there are no positive items for him
    if (sum(clicks_all[i][0:train_ind]) == 0):
        continue
    
    #skip user if there are no negative items for him
    if (sum(clicks_all[i][0:train_ind]) == len(clicks_all[i][0:train_ind])):
        continue

    random.seed(123) 
    neg_list_train = [random.choice(neg_items_train[i])]
    u_ = users[i]
    user_triplets = []
    for j in range(train_ind):

        if (clicks_all[i][j] == 1):
            p_ = items_for_user[i][j]

            X = {
                    model.user_ids: [u_]*(len(neg_list_train)),
                    model.left_ids: [p_]*(len(neg_list_train)),
                    model.right_ids: neg_list_train,
                    model.target_y: [1]*(len(neg_list_train))
                }
            user_triplets.append(X)       
        else:
            if (j>=1 and clicks_all[i][j-1] == 0):
                neg_list_train.append(items_for_user[i][j])

            else:
                neg_list_train = [items_for_user[i][j]]
    all_users_triplets.append(user_triplets)

    
start_time = time.time()
for epoch in range(200):
    for user_triplets in all_users_triplets:
        group_loss = 0
        for group in user_triplets:
            _ = model.session.run(model.train, feed_dict=group)
    
    if (time.time()-start_time > 3600):
        break
        
#Doing the predictions on 20% of items for each user
export_pred_user = open(export_basename + 'pr', 'w+')
export_true_user = open(export_basename + 'gt', 'w+')

for i_1 in range(len(users)):
    train_ind = int(train_part * len(items_for_user[i_1]))
    index = np.arange(train_ind, len(items_for_user[i_1]))
    pos_all_us = []


    for j in index:

        if (clicks_all[i_1][j] == 1):
            pos_all_us.append(items_for_user[i_1][j])

    items = items_for_user[i_1][train_ind:len(items_for_user[i_1])]

    if (len(pos_all_us)==0):
        continue

    if (len(items) !=1):
        fd = {
            model.user_ids:  (np.ones(len(items))*users[i_1]).astype(np.int32),
            model.left_ids: items

            }
        response = model.session.run(model.relevances, feed_dict=fd)


        # make relevances new pred
        itemsGroundTruth = pos_all_us
        predicted_ranking = np.argsort(-response)

        # write down predictions
        export_pred_user.write(' '.join(map(str, [users[i_1]] + list(np.array(items)[predicted_ranking]))) + '\n')
        export_true_user.write(' '.join(map(str, [users[i_1]] + list(itemsGroundTruth))) + '\n')



export_pred_user.close()
export_true_user.close()
output = subprocess.call(['bash','run_metrics.sh'])
