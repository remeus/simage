# simage

A script for Content-based Similar Image Retrieval.

![alt text](https://raw.githubusercontent.com/Remeus/simage/master/Images%20Retrieval.png "Example of images retrieval")

## Running the script
Training then testing of the model can be run directly using `python3 start.py`.

## Implementation
The code is composed of the different source files:
+ **start.py** The main file that performs training and testing of the model. By default training is
performed but it is possible to remove one line and uses the pretrained graph.
+ **main.py** The wrapper for training and testing. In the first case, the corresponding module
is called, in the second one the program checks for the existence of the pretrained model before
launching the testing on the computed transfer features.
+ **training.py** The main Tensorflow module for training, that computes the dataset, loads the
Tensorflow graph and runs a training session. By default the system works for a retrieval task of
60 images from the 1000 training images, but the arguments of the train_net function can be
changed easily to use the whole dataset and retrieve any number of images.
+ **inference.py** The Tensorflow module responsible for loading the pretrained model from the
working directory and using it to compute the output vector of a given image, which is then
used to output the most similar images. Once again the problem should be tested on 60 images
retrieved from 1000, as the full network could not fit for delivery. However it is possible to train
a network with the full output and run the testing by just updating the arguments of the infere
function.
+ **data_input.py** The helper module used to load the images and their corresponding ids, labels
and similarity vectors.
+ **similarity.py** The module used to create the previously mentioned similarity vectors that will
be used to assess the error of the network and backpropagate.
+ **check_score.py** A helper function that computes the f-score of the given training dataset based
on the similarity vectors. In other words, this is approximately the F-score targeted by our
network.
+ **inception_input.py** The module that performs the transfer of features between images and
inception output.
+ **inception.py, cache.py, download.py** Some helpers for the previous module.

## Network
The system is made of three main components:
1. A pretrained convolutional neural network, that computes feature vectors from the images.
2. A LSA/t-SNE module that embeds the labels into a dense space and computes "true" similarity
between training images.
3. A neural network that trains the images by attempting to reach the "true" similarity

![alt text](https://raw.githubusercontent.com/Remeus/simage/master/Network.png "Whole end-to-end network")

## Dataset
We use a subset of the [Google Open Images dataset](https://github.com/openimages/dataset), a set containing the URL of 9 million images together with their (mostly machine-generated) labelling.
Training is performed on 100,000 images of size 256 × 192, each with the corresponding couple of {label, confidence}.
Validation is performed on 10,000 images.

## Training
The training has been done on two GeForce GTX Titan GPUs. For 1000 images considered, it takes
approximately 5 minutes to train the model.
Above 30,000 images being considered, the longest part is clearly to pre-compute the "true" similarity
matrix. SVD manages still quite well, even with 100 000 images, but t-SNE is a slow algorithm and it
took 20 min to pre-compute the similarity matrix for 40,000 images. Fortunately this only needs to be
done once.
The rest of the training is performed quite fast based on the cache transfer features, and once the
similarity matrix is computed it does not take long to train it, even with 100,000 retrieval images.

## Validation
The overall F-score on validation, for the whole dataset, is 0.017, using 100 images being retrieved.
