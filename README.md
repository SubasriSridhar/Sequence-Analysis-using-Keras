## User Game Based Sequence Classification 

 

This experiments in this project aim to assess how predicting churn performs using sequential data to know whether the churn pattern lies in the sequential data. The experiments in this project is comparing Bi-directional LSTMs (a RNN approach) to RESNET(a CNN based approach) and measuring it’s performance. 

 

### 5.3.1. Bi-directional LSTMs

 

Recurrent neural network variants such as LSTMs are the one of the best options when modelling NLP-based problems, as they were expected to work very well on sequential data such as text. (Brownlee, 2016)

In RNNs, the output from each step is always sent as a input to the next step, therefore it recalls some information about the sequence. RNNs work well for a short sequence of data however, due to vanishing gradient problem, there are some drawbacks in recalling some longer sequence. To overcome this limitation, LSTM (Long Short-Term Memory) Networks, a improved versions of RNN is created which is versatile in recalling information over an long period using a gating mechanism that makes them selective in what previous information to be remembered and what not to remember and how much current input is to be added for building the current cell state. (Chollet, 2017)

![Image result for Bidirectional LSTM](file:///C:/Users/91979/AppData/Local/Temp/msohtmlclip1/01/clip_image002.jpg)

Figure 57. Bidirectional LSTM (i2tutorials, 2019)

 

Unidirectional LSTM only preserves information of the past because the inputs it has seen are from the past. Using bidirectional will run the inputs in two ways, one from past to future and one from future to past allowing it to preserve contextual information from both past and future at any point of time. (Vijay, 2019)

This is implemented in python:

\1.   The dataset is created by aggregating the sequence in one row per user. 

\2.   The dataset is loaded and encoded using one hot encoder

\3.   Then the data is split into test and train datasets. 

\4.   The sequence is limited to 20 and padded the input sequence with ‘0’ so that they are all of same length. The model trained in keras will know that zero carries no information.

\5.   Model architecture follows like this, first there is an embedding layer that learns the vector representation for each code followed by bidirectional LSTM. For regularization, Dropout is added as to prevent model over-fitting. (Dangeti, et al., 2017)

​        ![img](file:///C:/Users/91979/AppData/Local/Temp/msohtmlclip1/01/clip_image004.jpg)![img](file:///C:/Users/91979/AppData/Local/Temp/msohtmlclip1/01/clip_image006.jpg)

​                  Figure 58 . Bi-Directional LSTM Model         Figure 59 Model Architecture

The output layer i.e., softmax layer will give probability values for all the unique classes(1000) and based on the highest predicted probability, the model will classify the map name sequence to see whether the user churns or not. (Rai, 2019)

 

 

```
6.     The model is trained with 3 epochs, batch_size of 1024 and was able to achieve a loss of (0.9976) with 70%  accuracy for test data.
 

```

Figure 60. Sequence Analysis Result Summary

 

### 5.3.2. ResNet (Residual Networks)

 

This model uses residual blocks inspired from [ResNet](https://arxiv.org/abs/1512.03385) architecture. Deeper neural networks are difficult to train because of vanishing gradient problem — as the gradient is back-propagated to earlier layers, repeated multiplication may make the gradient infinitely small. As a result, as the network goes deeper, its performance gets saturated or even starts degrading rapidly. Res-Nets allows skip connection  which is adding initial input to the output of the convolution block. This eases the problem of diminishing gradient for gradient to flow through. It also allows the model to learn an identity function which ensures that the higher layer will perform at least as well as the lower layer, and not worse. (Vasilev, 2019)

There is also another concept called Dilated convolution which is introduced to convolution layers. So that the receptive field will be larger when compared with other ones. 

![img](file:///C:/Users/91979/AppData/Local/Temp/msohtmlclip1/01/clip_image009.jpg)

Figure 61. Dilated convolution

This defines a spacing between the values in a kernel. A 3x3 kernel with a dilation rate of 2 will have the same field of view as a 5x5 kernel, while only using 9 parameters. 

Implementation details are as follows: (Loy, 2019)

\1.   The dataset is created by aggregating the sequence in one row per user. 

\2.   The dataset is loaded and encoded using one hot encoder.

\3.   Then the data is split into test and train datasets. 

\4.   The sequence is limited to 20 and padded the input sequence with ‘0’ so that they are all of same length. The model trained in keras will know that zero carries no information.

\5.   Initial convolution operation is applied to the input with kernel size of 1 to extract basic properties.

\6.   Followed by that, Two same residual blocks were used to read patterns in data this is influenced by the ResNet design, which will enable to train the model with more epochs and improved output of simulations. ResNet is slightly different from the original design as this implementation uses two CNN and 1 dilation convolution instead of three CNN

![img](file:///C:/Users/91979/AppData/Local/Temp/msohtmlclip1/01/clip_image011.jpg)

Figure 62 Residual Block

\7.   Skip connection is introduced after convolution operations as input and output applied from convolution networks.

\8.   After residual blocks, Max pooling is applied for reducing the temporal size of representation.

\9.   Model architecture follows like this, first there is an embedding layer that learns the vector representation for each code followed by bidirectional LSTM. For regularization, Dropout is added as to prevent model over-fitting.

 

![img](file:///C:/Users/91979/AppData/Local/Temp/msohtmlclip1/01/clip_image013.jpg)

Figure 63. ResNet Model

 

 

 

 

 

![img](file:///C:/Users/91979/AppData/Local/Temp/msohtmlclip1/01/clip_image015.png)

Figure 64. RESNET Model Architecture

This model is trained with 3 epochs, batch_size of 1024 and validated on the validation data. (Ayyadevara, 2019)

![img](file:///C:/Users/91979/AppData/Local/Temp/msohtmlclip1/01/clip_image017.jpg)

Figure 65. Classification Report of ResNet

The results of this model are better than Bidirectional LSTM model. More improvements to this can be achieved by taking majority vote across an ensemble of models. (Vijay, 2019)

![img](file:///C:/Users/91979/AppData/Local/Temp/msohtmlclip1/01/clip_image019.jpg)

Figure 66. Confusion Matrix - ResNet

 

 

 

 

 

 

 

 

 

 

 

 