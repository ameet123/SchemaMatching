import os, numpy, csv
from minisom import MiniSom
import pickle, numpy
from scipy.spatial import distance
from collections import defaultdict
import dill
from sklearn.metrics import silhouette_score
from normalise import Normalise

# Training Data
dataFeaturePath = '../Feature_Vectors/DataFeatures_Train.pickle'
DataFeatures = pickle.load(open(dataFeaturePath, 'rb'))

# Test Data
testFeaturesPath = '../Feature_Vectors/DataFeatures_Match.pickle'
TestFeatures = pickle.load(open(testFeaturesPath,'rb'))


# List of all features in training and testing data
InitFeatures = []
all_attributes = []

for key,val in DataFeatures.items():
    all_attributes.append(key)
    InitFeatures.append(val)
    
for key,val in TestFeatures.items():
    all_attributes.append(key)
    InitFeatures.append(val)

# Normalise the Train + Test Features together 
c_features = Normalise(numpy.array(InitFeatures))

Features = []

for i in range(0,len(c_features[0])):
     #column = c_features[:][i]
     column = [row[i] for row in c_features]
     Features.append(column)


x = input('x attribute of grid')
y = input('y attribute of grid')
#Number of iterations to run

iteration = input("Input number of iterations: ")

#Create a SOM
som = MiniSom(x,y,20,sigma=0.3, learning_rate=0.5)
print "Training..."
som.train_random(Features, iteration) # trains the SOM with 100 iterations
print "...ready!"	

#Map the output neuron position to a unique cluster id. (0,0) --> 0, (0,1) --> 1 and so on.
feature_map = {}
k = 0

for i in range(x):
    for j in range(y):
        feature_map[(i,j)] = k
        k += 1
        
print feature_map, '\n'


      
#Write the attribute name and its corresponding id. DataFeatures is the dictionary (attribute, feature).
# Attribute_cluster is the dictionary (attribute, cluster_id). If the feature being mapped is same as the feature in the dictionary, save its winner ID in the dictionary.

attribute_cluster = defaultdict(lambda:defaultdict())

for i in range(0,len(all_attributes)):
     print feature_map[som.winner(Features[i])]
     attribute_cluster[feature_map[som.winner(Features[i])]][all_attributes[i]] = Features[i] 

'''    	
for k in DataFeatures.keys():
    for f in Features:
        if DataFeatures[k] == f:
            print feature_map[som.winner(f)]
            attribute_cluster[feature_map[som.winner(f)]][k] = f
            
#print attribute_cluster
for k in TestFeatures.keys():
    for f in Features:
        if TestFeatures[k] == f:
            attribute_cluster[feature_map[som.winner(f)]][k] = f

'''
print attribute_cluster

#pickle.dump(attribute_cluster,open('../Results/SOM_train_test_with_normal_49.pickle','w'))


#Open the output file for test data clusters
out_file_path_test = '../Results/Distances/SOM_with_normal_'+ str(x*y) + '_itr' + str(iteration) +'.csv'
out_file = open(out_file_path_test,'w')
output_file = csv.writer(out_file, delimiter = ',')
#output_file.writerow(["Attribute", "Cluster ID"])            

for key1,value in attribute_cluster.items():
    for key2 in value.keys():
        new_row = [key1, key2]
        output_file.writerow(new_row)

out_file.close()

labels = []
for f in Features:
    labels.append(feature_map[som.winner(f)])

silhouetteScore = silhouette_score(Features, labels)

print str(x*y), silhouetteScore

print 'Train Data: ', DataFeatures['tr_state']
print 'Test Data: ', TestFeatures['ts_state']

