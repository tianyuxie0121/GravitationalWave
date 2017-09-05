# GravitationalWave
My research work at UIUC.

1. Generating dataset to test and train from given dataset.
The given dataset is normalized to have total mass 1 and a variety of massratios. 
To create training and testing dataset, we have to generate signals with different total masses and massratios. 
What I do in generate_dataset is to read given dataset and to stretch the signal with times of the total mass.
Then I take the final 1 second of the stretched signal and down sample it to 8192Hz to get training signals and testing signals.
I use different sets of total masses in training and testing so they are totally seperate.

2. Creating noisy signals to test and train.
From generate_dataset we can get pure gravitational signals, but to simulate the real situation, we have to shift the signal and add noise to it. In function generator(), we can generate a banch of random shifted nosiy signals with given SNR. The shift and noise are both random, so we can avoid overfitting problem when we train our neural networks.

3. Neural Networks.
I'm using Tensorflow to build my neural networks. In order to get the highest accuracy to classify the extremely weak signals with the SNR=0.25, I 've tried a lot of structures, hyperparameters and activation functions. Basically I'm using covolution neural network which has 3 convolution layers and 2 fully connected layers. The original network in Daniel's paper is like this:
First it reshapes the input to be a tensor with the shape [-1,8192,1,1]. Then there is a convolution layer with the kernal shaped [16,1,1,16] and strides shaped [1,1,1,1]. Then there is a Relu layer. After that there is a pooling layer with the kernal shaped [1,4,1,1] and strides shaped [1,4,1,1]. Then comes the next convolution layer. Different with the first one, this one is dilated convolution layer with the kernal shaped [8,1,16,32] and rate=4, then comes the same Relu layer and pooling layer. After that there is another dilated convolution layer with the kernal shaped [8,1,32,64] and rate=4, the Relu layer and the pooling layer. Then the tensor will be flattened to have the shape [-1,7680]. Then the linear layer with the kernal shaped [7680,64] and the Relu layer. 
If we want to detect the existence of the gravitational wave, then the final two layers are the linear layer with the kernal shaped [64,2] and a Softmax layer.
If we want to predict the massratio from the gravitational wave, then the final two layers are the linear layer with the kernal shaped [64,1] and a Relu layer.
If we want to predict masses of two black holes from the gravitational wave, then the final two layers are the linear layer with the kernal shaped [64,2] and a Relu layer.
A loss function is required to compute the error after each iteration by measuring how close the outputs are with respect to the target values. Here we choose the mean squared error function for the predictor and the standard cross entropy loss for the classifier.

Here are some samples of networks I've tried.

1)CNNClassifier.
In this network I add batch normalization layer before each activation functions. The accuracy increases a little bit but not much. The network becomes unstable especially when testing with low SNR signals. 

2)CNNClassifier1.0.
In this network I replace all Relu layers with Elu layers. The difference between Relu and Elu layers is that when input x<0, elu(x)=a(exp(x)-1) while relu(x)=0. The network is better than the original one in accuracy and stability.

3)CNNClassifier2.0.
In this network I replace all Relu layers with Selu layers.
def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))
This network is no better than the original one.

4)CNNClassifier3.0.
In this network I add a dilated convolution layer with the kernal shaped [8,1,64,128] and rate=4 before the flatten layer. And the flatten layer should have the kernal shaped [-1,15360]. The network is better than the original one in accuracy. It seems that if we increase the output channel we can get better results.

5)CNNpredictor_massratio.
I reproduce Daniel's work of predicting the massratio of binary black holes. 

6)CNNpredictor_mass.
I reproduce Daniel's work of predicting the component masses of binary black holes.

As for training, we use ADAM method as our learning algorithm and start with learning rate=0.001.
I've tried different training method, such as to train with signals who have a range of SNR or fixed SNR. Training with fixed SNR can get better result at low SNR, but the network would forget information of the high SNR. Training with a range of SNR, say from x to 2, on the other hand, can always keep the network remembering the information of the high SNR, but the accuracy at low SNR is lower than training with fixed SNR. Also, I've tried to start with high SNR and gradually decrease the SNR, or start with low SNR and gradually increase the SNR. From my results the former seems to get better result. Remember when you train you shouldn't decrease the SNR until the cross entropy does not decrease for a time at the current SNR to get the best result.

To monitor the training process, I use Tensorboard to have a look of accuracy and cross entropy of both training and validation set. I monitor the logarithm of the cross entropy because it is too small to notice it's trend.

Results
For now the best result using fixed SNR is 85% success rate to detect noisy signals with SNR=0.2. If we use a range of SNR to train, then the success rate becomes 75% for noisy signals with SNR=0.2. In this way we can get more than 99% accuracy for SNR>=0.5. 
Our goal is 90% so still we need some improvements.
Note here SNR is defined to be the peak value of the pure signal divided by the standard deviation of the noise. 




