import tensorflow as tf


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
''' Testing Weights'''


''' 7 inputs neurons to 10 fully neurons Fully connected'''


Weights = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1],[-4 ,-3, -2, -1, -0.5, 0.5, 1, 2, 3, 4],[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
           [-0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07, -0.08, -0.09, -0.1],[0.31, -0.32, 0.43, 0.24,-0.85, 0.56,-0.47, 0.18,-0.99,-0.91],[0.31,-0.32, 0.43, 0.24,- 0.85, 0.56, 0.47, -0.18, 0.99,- 0.91]]

print(len(Weights))