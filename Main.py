#import tensorflow as tf
import copy
import math
import numpy as np
import matplotlib.pyplot as plt


import time


def make_Cone(a,b,c,radi,height,ax):
    choose=max(radi,height)
    # Set up the grid in polar
    theta = np.linspace(0,2*np.pi,90)
    r = np.linspace(0,choose,50)
    T, R = np.meshgrid(theta, r)

    # Then calculate X, Y, and Z
    X = R * np.cos(T) + a
    Y = R * np.sin(T) + b
    Z = (np.sqrt((X-a)**2 + (Y-b)**2)/(radi/height)) + c

    # Set the Z values outside your range to NaNs so they aren't plotted

    ax.plot_wireframe(X, Y, Z)


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
           [0.2, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1],
           [0.3 ,-3, -2, -1, -0, 0.5, 1, 2, 3, 4],
           [0.4, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
           [0.5, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07, -0.08, -0.09, -0.1],
           [0.6, -0.32, 0.43, 0.24,-0.85, 0.56,-0.47, 0.18,-0.99,-0.91],
           [0.7, 0.32, 0.43, 0.24,0.85, 0.56, 0.47, -0.18, 0.99,0.91]]

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

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def findMinProjection(xyDistances,zDistances,Layer1Z,angle):
    P = float('inf')
    for i in range(len(Layer1Z)):
        for j in range(len(Layer1Z)):
            if i!=j:
                try:
                    #line 1
                    A = [0, Layer1Z[i]]
                    B = [1/math.tan(angle[i]), Layer1Z[i]+1]

                    #line 2
                    C = [xyDistances[i][j], Layer1Z[j]]
                    D = [xyDistances[i][j] - 1/math.tan(angle[j]), Layer1Z[j]+1]

                    _,projection = line_intersection((A, B), (C, D))
                except:
                    projection = float('inf')
                if P > projection:
                    P = projection
                    iLow = i
                    jLow = j
    return P, iLow, jLow

def plotNext(angle,Projection,Layer1X,Layer1Y,Layer1Z,plot):
    radius = copy.deepcopy(angle)
    for i in range(len(radius)):
        specProg = Projection
        try: #DEAL WITH / 0
            radius[i] = specProg/radius[i]
        except:
            radius[i] = 0
    if plot == True:
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

    minProjection,iFound,jFound = findMinProjection(xyDistances,zDistances,Layer1Z,angle)

    #find radius due to projection and plot

    radius = plotNext(angle,minProjection,Layer1X,Layer1Y,Layer1Z,False)

    #find point of circle touching

    xCollision,yCollision,_,_ = get_intercetions(Layer1X[iFound], Layer1Y[iFound], radius[iFound], Layer1X[jFound], Layer1Y[jFound], radius[jFound])

    #find smaller angle remove and move remaining circle

    if angle[iFound] < angle[jFound]:    
        angle[iFound] = 0
        Layer1X[jFound] = xCollision
        Layer1Y[jFound] = yCollision
        Layer1Z[jFound] = minProjection
    else:
        angle[jFound] = 0
        Layer1X[iFound] = xCollision
        Layer1Y[iFound] = yCollision
        Layer1Z[iFound] = minProjection

    


    return  angle, Layer1X, Layer1Y, Layer1Z, last, radius, minProjection


keepGoing = True

import xlsxwriter

def output(filename, sheet, list1, list2, list3,list4,list5,list6):
    workbook   = xlsxwriter.Workbook(filename+'.xlsx')
    sh = workbook.add_worksheet()

    sh.write('A1', 'angle')
    sh.write('B1', 'Layer1X')
    sh.write('C1', 'Layer1Y')
    sh.write('D1', 'Layer1Z')
    sh.write('E1', 'radius')
    sh.write('F1', 'projection')
    sh.write_column('A2', list1)
    sh.write_column('B2', list2)
    sh.write_column('C2', list3)
    sh.write_column('D2', list4)
    sh.write_column('E2', list5)
    sh.write_column('F2', [list6])


    workbook.close()

dones = 0

def z_fun(x,y):
    return np.sin(np.sqrt(x**2+y**2))

def cone(x,y,a,b,c,R):
    return (np.sqrt((x-a)**2 + (y-b)**2))/R+c



while keepGoing == True:
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d') 
    byTheby = []
    byTheby.append(copy.deepcopy(Layer1X))
    byTheby.append(copy.deepcopy(Layer1Y))
    byTheby.append(copy.deepcopy(Layer1Z))
    byTheby.append(copy.deepcopy(angle))
    angle, Layer1X, Layer1Y, Layer1Z, keepGoing, radius, projection = findNeuronLocation(Layer1X,Layer1Y,Layer1Z,angle)
    byTheby.append(radius)
    byTheby.append(projection)
    
    height=projection

    for i in range(len(byTheby[0])):
        if byTheby[4][i] != 0:
            x=byTheby[0][i]
            y=byTheby[1][i]
            z=byTheby[2][i]
            radi=byTheby[4][i]
            make_Cone(x,y,z,radi,height,ax)
    
    print("OKI")

    plt.show()

print("OKAY")


