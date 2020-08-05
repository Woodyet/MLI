import tensorflow as tf


model = tf.keras.models.load_model('MainModel.h5') 

layersNweights = []

for layer in model.layers:
    layersNweights.append(layer.get_weights())






##############################
##############################
''' Testing Weights'''
