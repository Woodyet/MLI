#import tensorflow as tf
import copy
import numpy as np
import matplotlib.pyplot as plt

#model = tf.keras.models.load_model('MainModel.h5') 

#layersNweights = []

#for layer in model.layers:
#    layersNweights.append(layer.get_weights())

'''      MODEL
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
'''

##############################
##############################
def compareWeights(OrigWeights,Weights):
    x=0
    for compareLoop in OrigWeights:
        print(OrigWeights[x])
        print(Weights[x])
        print("-----")
        x+=1


''' Testing Weights'''


''' 7 inputs neurons to 10 fully neurons Fully connected'''


Weights = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
           [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1],
           [-4 ,-3, -2, -1, -0, 0.5, 1, 2, 3, 4],
           [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
           [-0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07, -0.08, -0.09, -0.1],
           [0.31, -0.32, 0.43, 0.24,-0.85, 0.56,-0.47, 0.18,-0.99,-0.91],
           [0.31,0.32, 0.43, 0.24,0.85, 0.56, 0.47, -0.18, 0.99,0.91]]

OrigWeights = copy.deepcopy(Weights)

#scale weights
maxWeight = float('-inf')
minWeight = float('inf')
for weightValuesPerNeuron in Weights:
    if max(weightValuesPerNeuron)>maxWeight:
        maxWeight = max(weightValuesPerNeuron)

    if min(weightValuesPerNeuron)<minWeight:
        minWeight = min(weightValuesPerNeuron)

for weightValuesPerNeuron in Weights:
    x=0
    for weightValue in weightValuesPerNeuron:
        if weightValue > 0:
            weightValuesPerNeuron[x] = weightValue/maxWeight
        elif weightValue < 0:
            weightValuesPerNeuron[x] = weightValue/(minWeight*-1)
        x+=1
            
#scaled.... Now apply activation function (RELU HERE)

for weightValuesPerNeuron in Weights:
    x = 0
    for weightValue in weightValuesPerNeuron:
        if weightValue < 0:
            weightValuesPerNeuron[x] =0
        x+=1

compareWeights(OrigWeights,Weights)

#assuming first layer draw input shape
#online code

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection


fig, ax = plt.subplots()
patches = []

polygon = RegularPolygon((0,0), numVertices=8, radius=1, orientation=0)
patches.append(polygon)

p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)

colors = 100*np.random.rand(len(patches))
p.set_array(np.array(colors))

ax.add_collection(p)

ax.set_xlim([-1.2,1.2])
ax.set_ylim([-1.2,1.2])

plt.show()


#Plotting a circle using complex numbers
#The idea: multiplying a point by complex exponential (enter image description here) rotates the point on a circle

num_pts=len(Weights) # number of points on the circle
ps = np.arange(num_pts)
# j = np.sqrt(-1)
pts = (np.exp(2j*np.pi/num_pts)**ps)
fig, ax = plt.subplots(1)
ax.plot(pts.real, pts.imag , 'o')
ax.set_aspect(1)
plt.show()