## Classification Models
The assignment involves implementation of three models of classification:
1. Fisher Discriminant Model (Linear Discriminant Method)
2. Probabilistic Generative Model
3. Logistic Regression (Discriminative Model)

#### Code Execution
For executing the code, first change the current directory to /A1/code/ of the repository. Compile the code using GCC:

	g++ main.cpp --std=c++11
	
Then execute using the following command:

	./a.out
	
### Results

**Fisher Discriminant Model**  

*Accuracy:*: 0.985437  
*Precision:* 0.983784  
*Recall:* 0.983784  

|               | Predicted = 0 | Predicted = 1  |
| ------------- | ------------- | -------------- |
| *Actual = 0*  | 	  224		    |		  3		       |
| *Actual = 1*  | 	  3	 	      |   	182		     |

**Probabilistic Generative Model**

*Accuracy:* 0.973301  
*Precision:* 0.943878  
*Recall:* 1  

|               | Predicted = 0 | Predicted = 1  |
| ------------- | ------------- | -------------- |
| *Actual = 0*  | 	  224		    |		  3		       |
| *Actual = 1*  | 	  3	 	      |   	182		     |

**Logistic Regression**

*Accuracy:*: 0.997573  
*Precision:* 1  
*Recall:* 0.994595  

|               | Predicted = 0 | Predicted = 1  |
| ------------- | ------------- | -------------- |
| *Actual = 0*  | 	  224		    |		  3		       |
| *Actual = 1*  | 	  3	 	      |   	182		     |

**NOTE**:  
The parameters used for the Gradient Descent algorithm in the task#3 are:  
* No. of iterations = 11000
* Learning Rate (alpha) = 0.003

