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


#compareWeights(WeightsPostAct,WeightsPostDistLog)

#for layer in NeuronAngle:
#    print(list(map(math.degrees,layer)))


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
Layer1Z = []

for _ in Weights:
    angle = math.radians(increment)
    Layer1X.append(math.sin(angle))
    Layer1Y.append(math.cos(angle))
    Layer1Z.append(0)
    increment+=360/len(Weights)

###Just a sanity check
'''
fig, ax = plt.subplots()
plt.scatter(Layer1X, Layer1Y)
ax.set_xlim([-1.2,1.2])
ax.set_ylim([-1.2,1.2])

plt.show()
'''
####Expand each point####
 
#set for testing 

#Layer1Z[4] = 0.01

#####

projectionToLayer2 = []

#just for first neuron for now


angle = []
for neuronAngles in NeuronAngle:
    try:  #DEAL WITH / 0
        angle.append(math.tan(neuronAngles[0]))
    except:
        angle.append(0)

#minAngleAboveZero = min(i for i in angle if i > 0)


#check... if undershoot then do again at higher projection (current + current/10)
def increaseProjection(Layer1X,Layer1Y,angle,initialProjection):
    radius = copy.deepcopy(angle)
    for i in range(len(radius)):
        try: #DEAL WITH / 0
            radius[i] = initialProjection/radius[i]
        except:
            radius[i] = 0
    res = checkForOverlappingCircles(Layer1X,Layer1Y,radius)
    if res == -1:
        initialProjection += initialProjection/10
        return radius,initialProjection,-1 #under
    elif res == 0:
        initialProjection -= initialProjection/7
        return radius,initialProjection,0 #over
    else:
        return radius,initialProjection,1 #perf


#-1 call increaseProjection
#0 call decreaseProjection
#1 stop

#next=-1

#while(next!=1):
#    radius,initialProjection,next = increaseProjection(Layer1X,Layer1Y,angle,initialProjection)


#for each point find distance to each other point in the x y
#also find distance in z

'''
eqn

L = ( d-w*math.tan(angle2) ) / (math.tan@1 + math.tan@2)

d = xy distance
w = z difference
angle 2 must be the higher neuron. ie check that z is > 0
L = projection required to have the cones touch.

'''


def findDistBetweenPoints(Layer1X,Layer1Y,Layer1Z):
    xyDistances = []
    zDistances = []
    for i in range(len(Layer1X)):
        eachxy = []
        eachz = []
        for j in range(len(Layer1X)):
            if i != j:
                x1 = Layer1X[i]
                x2 = Layer1X[j]
                y1 = Layer1Y[i]
                y2 = Layer1Y[j]
                z1 = Layer1Z[i]
                z2 = Layer1Z[j]

                xDiff = x1 - x2
                yDiff = y1 - y2
                zDiff = z1 - z2

                distD = math.sqrt(xDiff*xDiff+yDiff*yDiff)

                eachxy.append(distD)
                eachz.append(zDiff)
            if i == j:
                eachxy.append(-1)
                eachz.append(-1)

        xyDistances.append(eachxy)
        zDistances.append(eachz)
    return xyDistances,zDistances



def findMinProjection(xyDistances,zDistances,Layer1Z):
    P = float('inf')
    below = 0
    for i in range(len(xyDistances)):
        for j in range(len(xyDistances)):
            xyDist = xyDistances[i][j]
            zDist  = zDistances[i][j]
            if zDist == 0:
                aboveSameZ = xyDist * math.tan(angle[j]) * math.tan(angle[i])
                below = math.tan(angle[j]) + math.tan(angle[i])
                if below != 0:
                    calc = aboveSameZ / below
            else:
                if Layer1Z[i] < Layer1Z[j] and math.tan(angle[j]) > 0:
                    #aboveDiffZ = xyDist * math.tan(angle[j]) * math.tan(angle[i]) + abs(zDist)*math.tan(angle[i])
                    aboveDiffZ = math.tan(angle[j]) * math.tan(angle[i]) * (xyDist + abs(Layer1Z[j] - Layer1Z[i])/math.tan(angle[j]))
                    below = math.tan(angle[j]) + math.tan(angle[i])
                elif Layer1Z[j] < Layer1Z[i] and math.tan(angle[i]) > 0:
                    #aboveDiffZ = xyDist * math.tan(angle[j]) * math.tan(angle[i]) + abs(zDist)*math.tan(angle[j])
                    aboveDiffZ = math.tan(angle[j]) * math.tan(angle[i]) * (xyDist + abs(Layer1Z[i] - Layer1Z[j])/math.tan(angle[i]))
                    below = math.tan(angle[j]) + math.tan(angle[i])
                else:
                    below = 0
                if below != 0:
                    calc = aboveDiffZ / below
            if below != 0:
                if (calc < P and calc > 0):
                    P = calc
                    iLow = i
                    jLow = j      
    return P, iLow, jLow

def plotNext(angle,Projection,Layer1X,Layer1Y):
    radius = copy.deepcopy(angle)
    for i in range(len(radius)):
        try: #DEAL WITH / 0
            radius[i] = Projection/radius[i]
        except:
            radius[i] = 0
    fig, ax = plt.subplots()
    for i in range(len(Layer1X)):
        if radius[i] > 0:
            circle1 = plt.Circle((Layer1X[i], Layer1Y[i]), radius[i], color = 'r')
            ax.add_artist(circle1)
    ax.set_xlim([min(Layer1X)-1.2,max(Layer1Y)+1.2])
    ax.set_ylim([min(Layer1X)-1.2,max(Layer1Y)+1.2])
    plt.show()
    return radius

def get_intercetions(x0, y0, r0, x1, y1, r1):
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1

    d=math.sqrt((x1-x0)**2 + (y1-y0)**2)

    # non intersecting
    if d > r0 + r1 :
        return None
    # One circle within other
    if d < abs(r0-r1):
        return None
    # coincident circles
    if d == 0 and r0 == r1:
        return None
    else:
        a=(r0**2-r1**2+d**2)/(2*d)
        h=math.sqrt(r0**2-a**2)
        x2=x0+a*(x1-x0)/d   
        y2=y0+a*(y1-y0)/d   
        x3=x2+h*(y1-y0)/d     
        y3=y2-h*(x1-x0)/d 

        x4=x2-h*(y1-y0)/d
        y4=y2+h*(x1-x0)/d

        return x3, y3, x4, y4


def findNeuronLocation(Layer1X,Layer1Y,Layer1Z,angle):
    last = True
    #check if last calc
    temp = 0
    for item in angle:
        if item > 0:
            temp+=1
    
    if temp < 3:
        last = False

    #find distances

    xyDistances,zDistances = findDistBetweenPoints(Layer1X,Layer1Y,Layer1Z)

    #find smallest projection

    minProjection,iFound,jFound = findMinProjection(xyDistances,zDistances,Layer1Z)

    #find radius due to projection and plot

    radius = plotNext(angle,minProjection,Layer1X,Layer1Y)

    #find point of circle touching

    xCollision,yCollision,_,_ = get_intercetions(Layer1X[iFound], Layer1Y[iFound], radius[iFound], Layer1X[jFound], Layer1Y[jFound], radius[jFound])

    #find smaller angle remove and move remaining circle

    angle1 = angle[iFound]
    angle2 = angle[jFound]

    if angle1 < angle2:    
        angle[iFound] = 0
        Layer1X[jFound] = xCollision
        Layer1Y[jFound] = yCollision
        Layer1Z[iFound] = minProjection + Layer1Z[iFound]
    else:
        angle[jFound] = 0
        Layer1X[iFound] = xCollision
        Layer1Y[iFound] = yCollision
        Layer1Z[jFound] = minProjection + Layer1Z[jFound]

    


    return  angle, Layer1X, Layer1Y, Layer1Z, last


keepGoing = True
while keepGoing == True:
    angle, Layer1X, Layer1Y, Layer1Z, keepGoing = findNeuronLocation(Layer1X,Layer1Y,Layer1Z,angle)



print("OKAY")


