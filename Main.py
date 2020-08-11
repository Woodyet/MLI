#import tensorflow as tf
import copy
import math
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
'''
#scale weights #don't think needed#
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
'''

#apply activation fuction RELU in this case
for weightValuesPerNeuron in Weights:
    x = 0
    for weightValue in weightValuesPerNeuron:
        if weightValue < 0:
            weightValuesPerNeuron[x] = 0
        x+=1

WeightsPostAct = copy.deepcopy(Weights)

#apply 1/x Dist logic

WeightsPostDistLog = copy.deepcopy(WeightsPostAct)

for item in WeightsPostDistLog:
    x=0
    for weight in item:
        if weight == 0:
            item[x] = float('inf')
        else:
            item[x]=1/weight
        x+=1

#Find next neuron connection
#WeightsPostAct  #x
#WeightsPostDistLog #y

NeuronAngle = copy.deepcopy(WeightsPostDistLog)

i=0
for item in NeuronAngle:
    j=0
    for weight in item:
        if WeightsPostAct[i][j] > 0:
            if WeightsPostAct[i][j] < 1:
                NeuronAngle[i][j] = math.atan(WeightsPostAct[i][j]/WeightsPostDistLog[i][j])
            elif WeightsPostAct[i][j] > 1:
                NeuronAngle[i][j] = math.atan(WeightsPostDistLog[i][j]/WeightsPostAct[i][j])
            elif WeightsPostAct[i][j] == 1:
                NeuronAngle[i][j] = math.radians(45)
            else:
                input("ERROR IN INNER LOOP 1 VALUE" + str(WeightsPostAct[i][j]))

        elif WeightsPostAct[i][j] < 0:
            if WeightsPostAct[i][j] < -1:
                NeuronAngle[i][j] = math.atan((WeightsPostAct[i][j]/WeightsPostDistLog[i][j])*-1)*-1
            elif WeightsPostAct[i][j] > -1:
                NeuronAngle[i][j] = math.atan((WeightsPostDistLog[i][j]/WeightsPostAct[i][j])*-1)*-1
            elif WeightsPostAct[i][j] == -1:
                 NeuronAngle[i][j] = math.radians(45)*-1
            else:
                input("ERROR IN INNER LOOP 2 VALUE" + str(WeightsPostAct[i][j]))

        elif WeightsPostAct[i][j] == 0:
            NeuronAngle[i][j] = 0
        else:
            input("ERROR IN OUTER LOOP VALUE" + str(WeightsPostAct[i][j]))
        j+=1
    i+=1


compareWeights(WeightsPostAct,WeightsPostDistLog)

for layer in NeuronAngle:
    print(list(map(math.degrees,layer)))


#neuron angle is 

'''
               |  /
               | /
               |/  <------ This Angle
--------------------------------
 if neg-----> /|
             / |
            /  |

45 Degrees is the most "straight" 
'''
#assuming first layer draw input shape

increment=0
Layer1X = []
Layer1Y = []

for _ in Weights:
    angle = math.radians(increment)
    Layer1X.append(math.sin(angle))
    Layer1Y.append(math.cos(angle))
    increment+=360/len(Weights)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
plt.scatter(Layer1X, Layer1Y)
ax.set_xlim([-1.2,1.2])
ax.set_ylim([-1.2,1.2])

plt.show()