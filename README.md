This repo is focused on transforming the way neural networks are represented.

The standard case of...

.
 \__.
 /
.

Does not properly display the actual relationship between neurons in the network.

MRI provides doctors with a 3D model of the brain to identify issues, MLI hopes to achieve the same thing.

This is part of my PHD at QMUL and through experimentation I will attempt to identify 3D structures from neural networks that may give an 
insight into what to do to improve that specific neural network.

I hope to be able to identify such issues as Overfitting, the need for more training, redundant layers/neurons (and the opposite... i.e. the
need for more layers/neurons), unsuitable structures, incorrect learning rate. And potentially more.

I'll be adding work here as it progresses.

Eventually this will be an API request service where all that is needed to create a visual model will be

1. Number of layers
2. Type of layer
3. Size of layer
4. Activation functions used
5. weights/biases
