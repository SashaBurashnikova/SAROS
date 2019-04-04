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



def tf_cross_entropy(raw_margins, target_values, coefs, n_users, trunc_max=100):
    elementwise_entropy_loss = -tf.multiply(target_values,tf.log(raw_margins))-\
                                 tf.multiply(1-target_values,tf.log(1-raw_margins))
    checked_elwise_loss = tf.verify_tensor_all_finite(elementwise_entropy_loss, 
                                                      msg='NaN or Inf in loss vector', name='checked_elwise_ce')
    mean_loss = tf.reduce_sum(tf.multiply(tf.minimum(checked_elwise_loss, trunc_max), coefs))
    return mean_loss

def tf_mean_l2(w, coefs, n_users):
    elementwise_sq_norm = tf.reduce_sum(tf.pow(w, 2), axis=1)
    checked_elwise_l2 = tf.verify_tensor_all_finite(elementwise_sq_norm, msg='NaN or Inf in norm', name='checked_elwise_l2')
    mean_l2 = tf.reduce_sum(tf.multiply(checked_elwise_l2, coefs))
    return mean_l2


class BPR_MF(object):
    
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
            self.coefs_ = tf.placeholder(tf.float32, (None,), name='normalization_coefs')
            self.n_users = tf.placeholder(tf.float32, shape=(),  name='n_users_passed')
            self.a = 0.2
                                      
                                      
                                      
            # main parameters
            self.user_latents = tf.Variable(tf.random_uniform(shape=(self.N_USERS, self.N_EMBEDDINGS),seed=123),trainable=True, name='user_latents')
            self.item_latents = tf.Variable(tf.random_uniform(shape=(self.N_ITEMS, self.N_EMBEDDINGS),seed=124),trainable=True, name='item_latents')
                                      
                                      
            # get embeddings for batch
            self.embedding_user = tf.nn.embedding_lookup(self.user_latents,self.user_ids, name='embedding_user')
            self.embedding_left = tf.nn.embedding_lookup(self.item_latents,self.left_ids, name='embedding_left')
            self.embedding_right = tf.nn.embedding_lookup(self.item_latents,self.right_ids,name='embedding_right')
            self.embedding_mul = tf.multiply(self.embedding_left,self.embedding_user)
                                      
            # raw margins for primal ranking loss
            self.embedding_diff = self.embedding_left - self.embedding_right
            self.relevances = tf.reduce_sum(tf.multiply(self.embedding_user, self.embedding_left), axis=1)
                                      
            # shape: [n_batch, ]
            self.embedding_margins = tf.reduce_sum(tf.multiply(self.embedding_user, self.embedding_diff),axis=1, name='embedding_margins')
            self.embedding_loss = tf_cross_entropy(tf.sigmoid(self.embedding_margins),(self.target_y+1.)/2, self.coefs_ , self.n_users)
                                      
            # outs
            self.regularization = tf_mean_l2(self.embedding_user, self.coefs_, self.n_users) + \
                tf_mean_l2(self.embedding_left, self.coefs_, self.n_users) + \
                tf_mean_l2(self.embedding_right, self.coefs_, self.n_users)
            self.target_ = self.embedding_loss + self.alpha_reg * self.regularization
            
            self.opt = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
            self.train = self.opt.minimize(self.target_, var_list = [self.user_latents]) 
            self.target = self.target_ / self.n_users
            self.train2 = self.opt.minimize(self.target, var_list = [self.item_latents])
                                                                                                                                                                                                         
            self.init_all_vars = tf.global_variables_initializer()
                                                                                                                                                

    @property
    def weights_i(self):
        return self.item_latents.eval(session=self.session)
    
    @property
    def weights_u(self):
        return self.user_latents.eval(session=self.session)
    
    @property
    def get_ugrads_k(self):
        return [grad.eval(session=self.session) for grad in self.ugrads_k]
    
    @property
    def get_ugrads_l(self):
        return [grad.eval(session=self.session) for grad in self.ugrads_l]
    
    @property
    def get_igrads_k(self):
        return [grad.eval(session=self.session) for grad in self.igrads_k]
    
    @property
    def get_igrads_l(self):
        return [grad.eval(session=self.session) for grad in self.igrads_l]
    
    def initialize_session(self):
        config = tf.ConfigProto()
        # for reduce memory allocation
        config.gpu_options.allow_growth = True
        self.session = tf.Session(graph=self.graph, config=config)
        self.session.run(self.init_all_vars)
    
    def destroy(self):
        self.session.close()
        self.graph = None


latent_dim = 4
train_part = 0.8

################### New SAGA ##################
def get_negatives_and_positives_from_user(it_for_u, clicks):
    neg_all_us = []
    pos_all_us = []
    for n in range(len(clicks)):#saving all negative for each user for the next training
        
        if (clicks[n] == 0):
            neg_all_us = neg_all_us + [it_for_u[n]]
        
        if (clicks[n] == 1):
            pos_all_us = pos_all_us + [it_for_u[n]]
            
    return pos_all_us, neg_all_us

def build_all_triplets_for_loss(positives, negatives):
    neg_triplets = negatives*len(positives)
    pos_triplets = []
    for it in positives:
        pos_triplets += [it]*len(negatives)
    return pos_triplets, neg_triplets

# Read data
df = pd.read_csv('kassand_preprocessed.csv',sep=',',skiprows = 1, header = None)
#Extract the items for each user and create the list of clicked and non-clicked items for each user
users = set(df[0])
users = list(users)
items_for_user = []
neg_all_train = []
neg_all_test = []
pos_all_train = []
pos_all_test = []
us_list_train = []
us_list_test = []
num_items_all_test = 0
clicks_all = []
neg_all = []

for user in users:
    df_user  = df[df[0]==user].sort([3])#subdataset for each user, sorted by timestamp
    click = df_user[2]
    click = list(click)
    it_for_u = df_user[1]
    it_for_u = list(it_for_u)
    num_items_all_test = max(max(it_for_u), num_items_all_test)
    items_for_user.append(it_for_u)# list of items for each user
    clicks = []    
    for j in click:        
        if (j >= 1):
            clicks = clicks + [1]
        else:
            clicks = clicks + [0]
    
    clicks_all.append(clicks)
    train_ind = int(train_part*len(clicks))
    pos_us_train, neg_us_train = get_negatives_and_positives_from_user(it_for_u[:train_ind], clicks[:train_ind])
    pos_triplets, neg_triplets = build_all_triplets_for_loss(pos_us_train, neg_us_train)
    pos_all_train += pos_triplets
    neg_all_train += neg_triplets
    us_list_train += [user]*len(pos_triplets)
    neg_all.append(neg_us_train)
    
    
    pos_us_test, neg_us_test= get_negatives_and_positives_from_user(it_for_u[train_ind:], clicks[train_ind:])
    pos_triplets, neg_triplets = build_all_triplets_for_loss(pos_us_test, neg_us_test)
    pos_all_test += pos_triplets
    neg_all_test += neg_triplets
    us_list_test += [user]*len(pos_triplets)



#creating the files for saving results
export_basename = '/home/ama/burashna/icml_cross_entropy/data_saga/'


#find the number of users and items for the next part

num_users_all = max(users)+1
num_items_all_test += 1

#Restoring model for test part
model = BPR_MF(num_users_all, num_items_all_test, latent_dim, alpha_reg=0.01)
model.build_graph()
model.initialize_session()

# count number of triplets of each user in the sample
unq = np.unique(us_list_train, return_counts=True)
# count the number of unique users
n_users = len(unq[0])
# create the dictionary for coefficients (if we have 20 triplets for the 1st user, then coefficient for each triplet will be 0.05)
unq_dict = {unq[0][i]:1/float(unq[1][i]) for i in range(len(unq[0]))}
coefs = [unq_dict[i] for i in us_list_train]

# count number of triplets of each user in the sample
unq_1 = np.unique(us_list_test, return_counts=True)
# count the number of unique users
n_users_1 = len(unq_1[0])
# create the dictionary for coefficients (if we have 20 triplets for the 1st user, then coefficient for each triplet will be 0.05)
unq_dict_1 = {unq_1[0][i]:1/float(unq_1[1][i]) for i in range(len(unq_1[0]))}
coefs_1 = [unq_dict_1[i] for i in us_list_test]

Xtrain = {
        model.user_ids: us_list_train,
        model.left_ids: pos_all_train,
        model.right_ids: neg_all_train,
        model.target_y: [1]*(len(us_list_train)),
        model.coefs_ : coefs,
        model.n_users : n_users,
}

for i in range(30):
    _,_ = model.session.run([model.train, model.train2], feed_dict=Xtrain)
    
    #Doing the predictions on 20% of items for each user
    export_pred_user = open(export_basename + 'pr_b', 'w+')
    export_true_user = open(export_basename + 'gt_b', 'w+')

    for i_1 in range(len(users)):
        train_ind_all = int(train_part * len(items_for_user[i_1]))
        index = np.arange(train_ind_all, len(items_for_user[i_1]))
        pos_all_us = []


        for j in index:

            if (clicks_all[i_1][j] == 1):
                pos_all_us.append(items_for_user[i_1][j])

        items = items_for_user[i_1][train_ind_all:len(items_for_user[i_1])]

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
    output = subprocess.call(['bash','ml-1m_saga_b.sh'])
