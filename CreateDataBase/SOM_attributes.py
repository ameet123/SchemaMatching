import os, numpy, csv
from minisom import MiniSom
import pickle

dataFeaturePath = '../Codes/DataFeatures.pickle'

DataFeatures = pickle.load(open(dataFeaturePath, 'rb'))

Features = []
for val in DataFeatures.values():
    Features.append(val)

x = 7
y = 7

som = MiniSom(x,y,17,sigma=0.3, learning_rate=0.5)
print "Training..."
som.train_random(Features, 100) # trains the SOM with 100 iterations
print "...ready!"

feature_map = {}
k = 0

for i in range(x):
    for j in range(y):
        feature_map[(i,j)] = k
        k += 1
             

#print 'attribute			Spatial Position'
out_file_path = '../Codes/AttributeCluster_'+ str(x*y) +'.csv'
out_file = open(out_file_path,'w')
output_file = csv.writer(out_file, delimiter = ',')
output_file.writerow(["Attribute", "Cluster ID"])

print "number of rows to be written: ", len(Features)

i = 0

for k in DataFeatures.keys():
    for f in Features:
        if DataFeatures[k] == f:
            attribute_cluster = [k, feature_map[som.winner(f)]]
            output_file.writerow(attribute_cluster)

output_file.close()





