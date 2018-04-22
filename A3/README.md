## Handwriting Recognition (Using Neural Networks)

The objective is to perform handwriting recognition for digits 0-9. The dataset came from MNIST database of handwritten digits where each image is 28*28 square consisting of integers in range 0-255. The implementation has been done using Python3 and Keras.

The repository contains two implementations of neural networks:
1. Simple Artificial Neural Network
2. Convolutional Neural Network

Following contains the details:  
* The number of training examples available = 60000.
* For error calculation, cross entropy error function has been used.
* Mini-batch gradient descent is used with Adam optimizer, where **Batch_size = 228**.
* *Validation set error* is used to limit the number of iterations (epochs). The maximum number of iterations = 4000 (equivalently, 16 epochs here). The validation set is obtained by splitting the training data using the **split_ratio = 0.05**.
* Any combination of Convolution layer, Pooling layer, ReLu layer, Dropout and Flattening have been assumed to be acting as one hidden layer itself.
* For both the implementations, the number of layers have been varied between 3-5 to check for the best obtained results.


#### Code Execution

Before executing the code, make sure you have Python3 and Keras installed. For installing the required modules used, run the following command:

    $ sudo pip3 install -r requirements.txt

For executing the code, first change the current directory to A3/src/ of the repository. To run the code for simple neural network, type the following command:

    $ python3 simple_neural_network.py

Similarly, for executing the convolutional model, type the following in your terminal:
    
    $ python3 conv_neural_network.py

#### Results Obtained

For the artificial neural network, the following results were obtained:  
**Accuracy**: 0.9806 (3-layered network)  
**Number of Iterations**: 1500  

And, for the convolutional neural network, the following results were obtained:  
**Accuracy**: 0.9917 (5-layered network)  
**Number of Iterations**: 2000  

**NOTE**: The results may vary slightly from run to run. For reproducibility of results, a fixed seed is provided to the random number generator. Also, For the convolutional neural network, the code had to be run on a GPU due to the slow execution when using a CPU. 

