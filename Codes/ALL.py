"""
This run me file integrates one to one and many to one mappings.
"""
import csv
import pickle
import sys
from collections import defaultdict

import editdistance
import numpy as np
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity

from minisom import MiniSom
from normalise import Normalise

INFO = True
DEBUG = False


def log(log_str):
    if INFO:
        print log_str


def debug(log_str):
    if DEBUG:
        print log_str


'''
edit_distance is a map of string->dict
each key is an attribute or feature in test or train set.
'''
edit_distance = {}
euc_distance = {}
cosine_distance = {}
missing_attributes = []
# Global Dictionary mapping an attribute to possible smaller features (one --> many attributes)
Global_Dictionary = {('Name', 'PatientName'): ['First Name', 'First_Name', 'FName', 'F_Name', 'Last_Name', 'Last Name',
                                               'LName', 'L_Name'],
                     ('Address', 'Location', 'Addr', 'Residence', 'Loc'): ['Street Name', 'S_Name', 'St_Name',
                                                                           'Str_Name', 'Stree_Name', 'StName', 'St_No',
                                                                           'ST_Number', 'Street_No', 'S_No', 'S_Number',
                                                                           'Street Number', 'StNumber', 'StNo',
                                                                           'Apt_Num', 'Apartment_Number',
                                                                           'Apartment Number', 'Apartment No',
                                                                           'Apt_Number', 'Apt_No']}

def cosine_euc_edit_distance(testset_attributes, train_attributes):
    """
    Calculating cosine and euclidean distance of a test attribute from all train attributes.
    :param testset_attributes:
    :param train_attributes:
    :return:
    """
    for test_name in testset_attributes.keys():
        debug("Test feature: %s" % (test_name))
        edit_distance[test_name] = {}
        cosine_distance[test_name] = {}
        euc_distance[test_name] = {}

        if train_attributes == {}:
            log("\t>>Test attribute:%s missing from TrainSet" % test_name)
            cosine_distance[test_name]['None'] = 1
            edit_distance[test_name]['None'] = 1
            euc_distance[test_name]['None'] = 1
            missing_attributes.append(test_name)
        else:
            for name in train_attributes.keys():
                debug("\t>>Generating edit/cosine/euclidean distance for test attribute:%s to => TrainAttrib:%s" % (
                    test_name, name))
                edit_distance[test_name][name] = editdistance.eval(test_name, name)

                test_nd = np.asarray(testset_attributes[test_name]).reshape(1, -1)
                train_nd = np.asarray(train_attributes[name]).reshape(1, -1)
                cosine_distance[test_name][name] = cosine_similarity(test_nd, train_nd)[0][0]
                euc_distance[test_name][name] = distance.euclidean(testset_attributes[test_name],
                                                                   train_attributes[name])

    return cosine_distance, euc_distance, edit_distance


def cal_probability(distance_dic, name):
    """
    for each test attribute, we sum the values of all the distance for each corresponding train features.
    we calculate the probability as 1- <ind value>/total
    :param distance_dic:
    :param name:
    :return:
    """
    file_path = open('../Results/Distances/All_distances_combined/' + name + '.csv', 'w')
    Out = csv.writer(file_path, delimiter=',')

    new_row = ['test_attribute', 'train_attribute', 'distance', 'probability']
    Out.writerow(new_row)
    for test_attribute in distance_dic.keys():
        total = sum(distance_dic[test_attribute].values())
        for train_attribute in distance_dic[test_attribute].keys():
            if total == 0:
                prob = 1
            else:
                prob = 1 - (distance_dic[test_attribute][train_attribute] / float(total))
            new_row = [test_attribute, train_attribute, distance_dic[test_attribute][train_attribute], prob]
            Out.writerow(new_row)
    file_path.close()


def generate_many_relation(rel_type):
    """
    Generate one to many or many-to-one relation dictionary
    :param rel_type: OM:one to many or MO: many to one
    :return: dictionary of OM or MO relations
    """
    many_relation = {}
    flag = 0
    outer = Train_set if rel_type == 'OM' else Test_set
    inner = Test_set if rel_type == 'OM' else Train_set
    descriptive_string = "One-to-Many" if rel_type == "OM" else "Many-to-One"
    log(">>Processing %s relationships" % descriptive_string)

    for keys, values in Global_Dictionary.items():
        for k in keys:
            for attributes in outer:
                parent = 'tr_' + k
                debug(">>Comparing attrib:%s parent:%s" % (attributes, parent))
                if attributes.lower() == parent.lower():
                    flag = 1
                    many_relation[parent] = []
                    for inner_attrib in inner:
                        for val in values:
                            child = 'ts_' + val
                            if inner_attrib.lower() == child.lower():
                                log("\t>>%s has a possibility of adding to %s" % (attributes, inner_attrib))
                                many_relation[parent].append(child)

    if flag == 0: print "no %s possible" % descriptive_string
    return many_relation


def generate_one_to_one_features():
    """
    list of one to one features = All Train features - The Many-relation features
    :return: one to one feature list
    """
    OnetoMany = generate_many_relation('OM')
    ManytoOne = generate_many_relation('MO')
    log(">>Generating one to one feature list by subtracting the manies relations from ALL train set")
    the_manies = OnetoMany.keys() + ManytoOne.keys()
    one_to_one_features = [x.lower() for x in Train_set + Test_set if x not in the_manies]
    return one_to_one_features


def normalized_one_to_one_attribs(one_to_one_attribs):
    """
    Build list of one to one feature values
    we do this by building a list of relevant attributes and then transposing the columns into rows using numpy
    :param one_to_one_attribs:
    :return:
    """
    train_test_items = DataFeatures.copy()
    train_test_items.update(TestFeatures)

    InitFeatures = [train_test_items[x] for x in one_to_one_attribs]
    log(">>Train+Test dictionary size:%d One-to-One feature values size:%d" % (
        len(train_test_items), len(InitFeatures)))

    c_features = Normalise(np.array(InitFeatures))

    log(">>c_features type:%s size:%d len of [0]:%d" % (type(c_features), len(c_features), len(c_features[0])))
    debug(c_features)
    # Transponse into an array with elements equal to number of features.
    normalized_features = np.asarray(c_features).T
    log("=== Size of Features used in SOM: %d" % len(normalized_features))
    return normalized_features


def neuron_feature_map(m, n):
    """
    Map the output neuron position to a unique cluster id. (0,0) --> 0, (0,1) --> 1 and so on.
    :return:
    """
    fm = {}
    k2 = 0
    for i1 in range(m):
        for j in range(n):
            fm[(i1, j)] = k2
            k2 += 1
    return fm

def som_winners_attrib_cluster(m, n):
    """
    Write the attribute name and its corresponding id. DataFeatures is the dictionary (attribute, feature).
    attribute_cluster is the dictionary (attribute, cluster_id).
    If the feature being mapped is same as the feature in the dictionary, save its winner ID in the dictionary.
    :param m: 
    :param n: 
    :return: 
    """
    som_model = MiniSom(m, n, 20, sigma=0.3, learning_rate=0.5)
    log("Training...[%d,%d] for iterations=%d" % (m, n, iteration))
    som_model.train_random(Features, iteration)  # trains the SOM with 100 iterations
    log("...ready!")
    feature_map = neuron_feature_map(m, n)
    log(">>Generating attribute clusters with corresponding cluster id")
    attribute_clusters_dict = defaultdict(dict)
    for i in range(0, len(all_attributes)):
        attribute_clusters_dict[feature_map[som_model.winner(Features[i])]][all_attributes[i]] = Features[i]

    pickle.dump(attribute_clusters_dict, open('../Results/17_FinalSOM_with_normal_' + str(int(m * n)) + '.pickle', 'w'))
    return attribute_clusters_dict, som_model


def generate_all_distances(attribute_clusters_map):
    global cosine_distance, euc_distance, edit_distance
    log(">>Generating cosing/euclidean/edit distances for test and train features by first splitting attributes")
    for cluster_id in attribute_clusters_map.values():
        test_names_features = {}
        train_names_features = {}

        for attribute in cluster_id.keys():
            if attribute in Test_set:
                test_names_features[attribute] = cluster_id[attribute]
            else:
                train_names_features[attribute] = cluster_id[attribute]
        cosine_distance, euc_distance, edit_distance = cosine_euc_edit_distance(test_names_features,
                                                                                train_names_features)

if __name__ == '__main__':
    log(">>Running SOM and calculating distance probabilities.")
    if len(sys.argv) != 4:
        raise ValueError('Please provide x, y, iterations')

    x = int(sys.argv[1])
    y = int(sys.argv[2])
    iteration = int(sys.argv[3])
    log("x = %d y=%d iteration=%d" % (x, y, iteration))

    #########
    log("1. Loading Data")
    dataFeaturePath = '../Feature_Vectors/DataFeatures_Train.pickle'
    testFeaturesPath = '../Feature_Vectors/DataFeatures_Match.pickle'

    # Loading Data
    DataFeatures = pickle.load(open(dataFeaturePath, 'rb'))
    TestFeatures = pickle.load(open(testFeaturesPath, 'rb'))

    Train_set = [k for k in DataFeatures.keys()]
    Test_set = [k for k in TestFeatures.keys()]
    #########
    log("2. Generating features and normalizing")
    all_attributes = generate_one_to_one_features()
    Features = normalized_one_to_one_attribs(all_attributes)
    log("====Remaining Features=========: %d normalized transposed list:%d" % (len(all_attributes), len(Features)))
    #########
    log("3. Generating SOM Model and distances")
    attribute_clusters, som = som_winners_attrib_cluster(x, y)
    generate_all_distances(attribute_clusters)
    #########
    log("4. Calculating probabilities and storing results")
    cal_probability(edit_distance, 'FinalSOM_edit_' + str(x * y))
    cal_probability(cosine_distance, 'FinalSOM_cosine_' + str(x * y))
    cal_probability(euc_distance, 'FinalSOM_euc_' + str(x * y))
    #########
