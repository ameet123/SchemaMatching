#!/usr/bin/env python
# coding: utf-8

# In[47]:


import csv
import operator
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
from minisom import MiniSom
import editdistance
from scipy.spatial import distance


# In[48]:


import os
os.chdir("C:\\Users\\ameet.chaubal\\Documents\\source\\Schema-Matching-using-Machine-Learning")


# In[49]:


x=5
y=5
iterations=1000
normalized_yn="N"
edit_distance = {}
cosine_distance = {}
euc_distance = {}


# In[50]:


if normalized_yn=="N":
    dataFeaturePath = './Feature_Vectors/DataFeatures_Train.pickle'
    test_featurePath = './Feature_Vectors/DataFeatures_Match.pickle'
else:
    dataFeaturePath = './Feature_Vectors/normalised_features_train.pickle'
    test_featurePath = './Feature_Vectors/normalised_features_match.pickle'


# In[51]:


DataFeatures = pickle.load(open(dataFeaturePath, 'rb'))
TestFeatures = pickle.load(open(test_featurePath, 'rb'))
dims=len(list(DataFeatures.values())[0])


# In[52]:


def convert_to_np(dict_vals):
    f = list(dict_vals.values())
    print ("Rows:{} cols:{}".format(len(f),len(f[0])))
    n_f = np.array(f)
    print (">>np features:{}".format(n_f.shape))
    return n_f


# In[53]:


np_train_features = convert_to_np(DataFeatures)
np_test_features = convert_to_np(TestFeatures)


# ### Choose normalized or raw data

# In[54]:


if normalized_yn=="N":
    normalized_feat = np_train_features
else:
    min_max_scaler = preprocessing.MinMaxScaler()
    normalized_feat = min_max_scaler.fit_transform(np_train_features)
    print (">>normalized features:{}".format(normalized_feat.shape))


# In[55]:


# Combine keys and normalized features into dict
normalized_features_dict=dict(zip(DataFeatures.keys(),normalized_feat))
print ("Normalized features: rows={} cols={}".format(len(normalized_features_dict),len(normalized_features_dict['tr_hcp_footnote'])))


# In[56]:


som = MiniSom(x, y, dims, sigma=0.3, learning_rate=0.5)
print ("Training...")
som.train_random(normalized_feat, iterations,verbose=True)
print ("...ready!")


# #### Predicting Cluster ID for each feature key & generating map of cluster ID => list of feature value vectors

# In[57]:


attribute_cluster_map={}
cluster_data_vector_map=defaultdict(list)
cluster_attrib_map=defaultdict(list)
predicted_clusters=[]
for key,data in normalized_features_dict.items():
    winid = som.winner(data)
    clusterid=np.ravel_multi_index(winid,(x,y))
    predicted_clusters.append(clusterid)
    attribute_cluster_map[key]=clusterid
    cluster_attrib_map[clusterid].append(key)
    cluster_data_vector_map[clusterid].append(data)


# In[58]:


print("Total uniq clusters:{}".format(len(cluster_data_vector_map)))
cluster_center={}
for k,v in cluster_data_vector_map.items():
    center  = [sum(j)/len(v) for j in zip(*v)]
    cluster_center[k]=center


# In[59]:


for k,v in cluster_attrib_map.items():
    print("{} =>{}".format(k,v))


# #### Finding a match for each Test feature

# In[60]:


test_attrib_clusterid_map={}
for k,v in TestFeatures.items():
    eudistance = []
    min_dist = 9000000
    for centerID, center in cluster_center.items():
        eudistance.append(distance.euclidean(v, center))
        min_d = min(eudistance)
        if min_d < min_dist:
            min_dist = min_d
            test_attrib_clusterid_map[k] = centerID


# #### Map of test attribute and corresponding train attributes possibility

# In[61]:


for k,v in test_attrib_clusterid_map.items():
    print ("{} => {}:{}\n".format(k,v, cluster_attrib_map[v]))


# In[62]:


sil_score= silhouette_score(list(DataFeatures.values()),predicted_clusters)


# In[63]:


def calculate_edit_distance(test_name, train_names_features):
    global edit_distance
    edit_distance[test_name] = {}
    for name in train_names_features:
        edit_distance[test_name][name] = editdistance.eval(test_name, name)


# In[64]:


def cosine_euc_distance(test_name, test_feature, train_name_features):
    # Calculating cosine and euclidean distance of a test attribute from all train attributes. 

    cosine_distance[test_name] = {}
    euc_distance[test_name] = {}
    for name in train_name_features.keys():
        # print "shape of test_features:%s train_name_features:%s => name:%s" % test_feature.shape, train_name_features[
        #     name].shape, name
        test_nd = np.asarray(test_feature).reshape(1, -1)
        train_nd = np.asarray(train_name_features[name]).reshape(1, -1)
        cosine_distance[test_name][name] = cosine_similarity(test_nd,train_nd)[0][0]
        euc_distance[test_name][name] = distance.euclidean(test_feature, train_name_features[name])


# In[65]:


def cal_probability(distance_dic, dic2, dic3):
    prob_list=[]
    file_path = open('./Results/avg_prob__'+str(x)+"_"+str(y)+".csv", 'w')
    Out = csv.writer(file_path, delimiter=',')
    new_row = ['test_attribute', 'train_attribute', 'distance', 'avg_probability']
    Out.writerow(new_row)
    for test_attribute in distance_dic.keys():
        total = sum(distance_dic[test_attribute].values())
        total2 = sum(dic2[test_attribute].values())
        total3 = sum(dic3[test_attribute].values())

        for train_attribute in distance_dic[test_attribute].keys():
            prob = 1 - (distance_dic[test_attribute][train_attribute] / float(total))
            prob2 = 1 - (dic2[test_attribute][train_attribute] / float(total2))
            prob3 = 1 - (dic3[test_attribute][train_attribute] / float(total3))
            avg_dist = (distance_dic[test_attribute][train_attribute]+
                        dic2[test_attribute][train_attribute]+dic3[test_attribute][train_attribute])/float(3)
            avg_prob = (prob+prob2+prob3)/float(3.0)
            new_row = [test_attribute, train_attribute, avg_dist, avg_prob]
            prob_list.append(new_row)
            Out.writerow(new_row)
    file_path.close()
    return prob_list


# In[66]:


print ("test_attrib size:{} train attrib_clus size:{}".
       format(len(test_attrib_clusterid_map), len(attribute_cluster_map)))  
for key, val in test_attrib_clusterid_map.items():
    train_names = []
    train_name_features = {}
    for k, v in attribute_cluster_map.items():
        if val == v:
            train_names.append(k)
            train_name_features[k] = DataFeatures[k]
    print("for key:{} \ntrainnames:{}\ntrain_feature:{}".
          format(key,train_names,train_name_features))
    calculate_edit_distance(key, train_names)
    cosine_euc_distance(key, TestFeatures[key], train_name_features)


# In[67]:


print(edit_distance)


# #### Calculate distance based probabilities

# In[68]:


# edit_dist_prob   = cal_probability(edit_distance )
# cosine_dist_prob = cal_probability(cosine_distance)
# eucl_dist_prob   = cal_probability(euc_distance)
avg_prob = cal_probability(edit_distance, cosine_distance, euc_distance)


# ### Results

# In[69]:


print("Cluster:[{},{}]\n{}\nTotal Uniq Clusters:{}\nSilhouetteScore:{}".
      format(x,y,"="*20,len(cluster_data_vector_map),sil_score))

