## Neural Networks Implementation

The objective is to implement a neural network to classify handwritten digits from 0-9. The details of the model are as follows:

* The network is 3-layered: number of units being 64, varying b/w 5 to 10 and 10 respectively for the input, hidden and output layers. 
* Weights have been initialized randomly b/w -1 and 1.
* Softmax function has been used for the error calculation.
* Gradient descent is done in mini batch mode with the batch size set to 100.
* The implementation performs momentum update using adaptive learning, and used Adam's optimizer method for the same. The parameters used for the same are as follows:
    * **Alpha:** 0.01
    * **Beta1:** 0.9
    * **Beta2:** 0.999
    * **epsilon:** 10e-8
* For convergence, the maximum number of iterations is set to 3000. But, the validation data is used to stop the descent in between if the error increases on the validation data.

#### Code Execution
For executing the code, first change the current directory to A2/ of the repository. Compile the code using GCC:

	g++ main.cpp
	
Then execute using the following command:

	./a.out

#### Results Obtained

The table below contains the results after varying the number of hidden units b/w 5 to 10.

|  Number of Hidden Units   | Accuracy 		| Number of Iterations  |
| --------------------------| ------------- | --------------------- |
| 		  		5			| 	0.852695   	|		  1285	     	|
| 				6		    |   0.867665	|		  1059			|
|				7			|	0.872455	|		  1263			|
|				8			|	0.905988	|		  1107			|
|				9			|	0.936527	|		  1306			|
|				10			|	0.937126	|		  1267			|

**NOTE**: The results vary from run to run as the weights are initialized randomly. 