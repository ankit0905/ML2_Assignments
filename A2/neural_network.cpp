/* Class to implement Artificial Neural Network - specifically
    for the problem of handwriting recognition.

    attributes:
    @training_data: set of training examples
    @validation_data: data for validation

    @weights_kj: weight vectors b/w hidden and output layers
    @weights_ji: weight vectors b/w hidden and input layers
    @d_E_wkj: derivates wrt weights b/w hidden and output layers
    @d_E_wji: derivates wrt weights b/w hidden and input layers

    @T_k: target values
    @X_i: values of the input units
    @Z_j: activated values of the hidden units

    @delta_k: error for the output layer units
    @delta_j: error for the hidden layer units

    @mt_kj: First order moment for derivatives wrt weights b/w hidden and output layers
    @mt_kj_corrected: Corrected biased first order moments for mt_kj values
    @mt_ji: First order moment for derivatives wrt weights b/w hidden and input layers
    @mt_ji: Corrected biased first order moments for mt_ji values
    @vt_kj: Second order moment for derivatives wrt weights b/w hidden and output layers
    @vt_kj_corrected: Corrected biased first order moments for vt_kj values
    @vt_ji: Second order moment for derivatives wrt weights b/w hidden and input layers
    @vt_ji_corrected: Corrected biased first order moments for vt_ji values

    @temp_kj, @temp_ji: Temporary set of vectors for holding weights before the actual updation

    @A_j: hidden layer unit values before activation
    @A_k: output layer unit values before activation
    @Y_k: predicted values for output layer nodes using activation of A_k values

    @D: input dimensions (or, number of input units excluding bias)
    @N: number of training examples
    @M: number of hidden layer nodes excluding bias
    @K: output dimensior (or, number of output layer nodes)

    @index: variable to keep track the index of training example to be trained
    @t: variable to keep track of time (used in momentum update as exponent for beta values)

    CONSTANTS
    @batch_size: size of the mini batch of to be trained in one go
    @beta1, @beta2, @epsilon: hyperparameters for gradient descent (part of Adam's optimizer)
*/
class NeuralNetwork
{
    private:
        vector<pair<vector<double>, int> > training_data, validation_data;
        vector<vector<double> > weights_ji, weights_kj, T_k, d_E_wkj, d_E_wji,
             X_i, Z_j, delta_k, delta_j, mt_kj_corrected, mt_kj, vt_kj_corrected, vt_kj,
             mt_ji_corrected, mt_ji, vt_ji_corrected, vt_ji, temp_kj, temp_ji;
        vector<double> A_j, A_k, Y_k;
        int D, N, M, K, index = 0, t = 1;
        const int batch_size = 100;
        const double beta1 = 0.9, beta2 = 0.999, epsilon = 10e-8;

    public:
        NeuralNetwork(vector<pair<vector<double>, int> > train_data, 
            vector<pair<vector<double>, int> > validation_data, int no_hidden_units);
        double sigmoid(double value);
        void generateTargetValues();
        void initialize();
        void initializeWeights();
        void forwardPropagation();
        void backpropagation();
        double calculateError();
        void runGradientDescent(double alpha, int num_iterations);
        double predict(vector<pair<vector<double>, int> > train_data);
};

/* Constructor method to do all the necessary initialiations such as
    getting the data and setting up the parameters.

    params:
    @train_data: training data of handwritten digits
    @validation_data: data for validation while running gradient descent
    @no_hidden_units: number of units in hidden layer (default=5)
*/
NeuralNetwork::NeuralNetwork(vector<pair<vector<double>, int> > train_data, 
    vector<pair<vector<double>, int> > validation_data, int no_hidden_units=10)
{
    training_data = train_data;
    this -> validation_data = validation_data;
    
    D = train_data[0].first.size();
    N = train_data.size();
    M = no_hidden_units;
    K = 10;

    initialize();
    initializeWeights();
    generateTargetValues();
}

/* Method that allocates space for various vector variables used
    within the implementation.
*/
void NeuralNetwork::initialize()
{
    for(int k=0; k<K; k++){
        Y_k.push_back(0);
        A_k.push_back(0);
        vector<double> row;
        for(int j=0; j<=M; j++) row.push_back(0);
        d_E_wkj.push_back(row);
        mt_kj.push_back(row);
        mt_kj_corrected.push_back(row);
        vt_kj.push_back(row);
        vt_kj_corrected.push_back(row);
        temp_kj.push_back(row);
    }
    for(int j=0; j<M; j++){
        A_j.push_back(0);
        vector<double> row;
        for(int i=0; i<=D; i++) row.push_back(0);
        d_E_wji.push_back(row);
        mt_ji.push_back(row);
        mt_ji_corrected.push_back(row);
        vt_ji.push_back(row);
        vt_ji_corrected.push_back(row);
        temp_ji.push_back(row);
    }

    for(int j=0; j<batch_size; j++){
        vector<double> row;
        for(int i=0; i<=D; i++) row.push_back(0);
        X_i.push_back(row);
    }

    for(int j=0; j<batch_size; j++){
        vector<double> row;
        for(int i=0; i<=M; i++) row.push_back(0);
        Z_j.push_back(row);
    }

    for(int j=0; j<batch_size; j++){
        vector<double> row;
        for(int i=0; i<M; i++) row.push_back(0);
        delta_j.push_back(row);
    }
    
    for(int j=0; j<batch_size; j++){
        vector<double> row;
        for(int i=0; i<K; i++) row.push_back(0);
        delta_k.push_back(row);
    }
}

/* Method to return sigmoid of a value 

    params:
    @value: input to function

    returns:
     The sigmoid value
*/
double NeuralNetwork::sigmoid(double value)
{
    return 1.0/(1.0+exp(-value));
}

/* Method to generate the target values for the training data and 
    insert them to corresponding vector.
*/
void NeuralNetwork::generateTargetValues()
{
    for(int i=0; i<N; i++){
        vector<double> row;
        for(int j=0; j<K; j++) row.push_back(0);
        T_k.push_back(row);
        T_k[i][training_data[i].second] = 1;
    }
}

/* Method to randomly initialize all the weight parameters of the neural network.
    There are two sets of weights: output layer to hidden layer and hideen layer
    to input layer.
*/
void NeuralNetwork::initializeWeights()
{
    srand(time(NULL));
    
    for(int j=0; j<M; j++){
        vector<double> row;
        for(int i=0; i<=D; i++) row.push_back(2*((double)rand()/RAND_MAX)-1);
        weights_ji.push_back(row);
    }

    for(int k=0; k<K; k++){
        vector<double> row;
        for(int j=0; j<=M; j++) row.push_back(2*((double)rand()/RAND_MAX)-1);
        weights_kj.push_back(row);
    }
}

/* Method to perform feed forward propagation and uses mini batch mode where 
    the batch size is set to 100.
*/
void NeuralNetwork::forwardPropagation()
{
    int counter = 0;
    while(counter < batch_size){
        for(int i=0; i<D; i++) X_i[counter][i] = training_data[index].first[i];
        X_i[counter][D] = 1;
        
        for(int j=0; j<M; j++){
            A_j[j] = 0;
            for(int i=0; i<=D; i++){
                A_j[j] += weights_ji[j][i] * X_i[counter][i];
            }
            Z_j[counter][j] = sigmoid(A_j[j]);
        }
        Z_j[counter][M] = 1;

        for(int k=0; k<K; k++){
            A_k[k] = 0;
            for(int j=0; j<=M; j++){
                A_k[k] += weights_kj[k][j] * Z_j[counter][j];
            }
            Y_k[k] = sigmoid(A_k[k]);
            delta_k[counter][k] = (Y_k[k] - T_k[index][k]);
        }

        index = (index + 1)%N;
        counter++;
    }
}

/* Method that implements backpropagation algorithm. 
*/
void NeuralNetwork::backpropagation()
{
    for(int i=0; i<batch_size; i++){
        for(int j=0; j<M; j++){
            double sum = 0;
            for(int k=0; k<K; k++){
                sum += weights_kj[k][j] * delta_k[i][k];
            }
            delta_j[i][j] = sum * Z_j[i][j] * (1 - Z_j[i][j]);
        }
    }

    for(int k=0; k<K; k++){
        for(int j=0; j<=M; j++){
            d_E_wkj[k][j] = 0;
            for(int c=0; c<batch_size; c++){
                d_E_wkj[k][j] += delta_k[c][k] * Z_j[c][j];
            }
            d_E_wkj[k][j] /= batch_size;

            mt_kj[k][j] = beta1 * mt_kj[k][j] + (1 - beta1) * d_E_wkj[k][j];
            mt_kj_corrected[k][j] = mt_kj[k][j] / (1 - pow(beta1, t));
            vt_kj[k][j] = beta2 * vt_kj[k][j] + (1 - beta2) * pow(d_E_wkj[k][j],2);
            vt_kj_corrected[k][j] = vt_kj[k][j] / (1 - pow(beta2, t));
        }
    }

    for(int j=0; j<M; j++){
        for(int i=0; i<=D; i++){
            d_E_wji[j][i] = 0;
            for(int c=0; c<batch_size; c++){
                d_E_wji[j][i] += delta_j[c][j] * X_i[c][i];
            }
            d_E_wji[j][i] /= batch_size;

            mt_ji[j][i] = beta1 * mt_ji[j][i] + (1 - beta1) * d_E_wji[j][i];
            mt_ji_corrected[j][i] = mt_ji[j][i] / (1 - pow(beta1, t));
            vt_ji[j][i] = beta2 * vt_ji[j][i] + (1 - beta2) * pow(d_E_wji[j][i],2);
            vt_ji_corrected[j][i] = vt_ji[j][i] / (1 - pow(beta2, t));
        }
    }
}

/* Method that Calculates the error on the validation data for the current set of 
    weight parameters. Softmax function has been used. This plays a crucial role in
    limiting the number of iterations of gradient descent.

    returns:
    @total: the error calculated (absoluted value returned)
*/
double NeuralNetwork::calculateError()
{
    int sz = validation_data.size(), n = 0;

    vector<vector<double> > target_values;
    vector<double> input(D+1);

    for(int i=0; i<sz; i++){
        vector<double> row;
        for(int j=0; j<K; j++) row.push_back(0);
        target_values.push_back(row);
        target_values[i][validation_data[i].second] = 1;
    }

    double total = 0;

    while(n < sz){
        for(int i=0; i<D; i++) input[i] = validation_data[n].first[i];
        input[D] = 1;

        for(int j=0; j<M; j++){
            A_j[j] = 0;
            for(int i=0; i<=D; i++){
                A_j[j] += weights_ji[j][i] * input[i];
            }
            Z_j[0][j] = sigmoid(A_j[j]);
        }

        Z_j[0][M] = 1;
        for(int k=0; k<K; k++){
            A_k[k] = 0;
            for(int j=0; j<=M; j++){
                A_k[k] += weights_kj[k][j] * Z_j[0][j];
            }
            Y_k[k] = sigmoid(A_k[k]);
        }

        for(int k=0; k<K; k++){
            total += target_values[n][k]*log(Y_k[k]) 
                        + (1-target_values[n][k])*log(1-Y_k[k]);
        }
        n++;
    }
    return abs(total);
}

/* The main method to run gradient descent algorithm for the neural network.
    It first performs feed forward prpagation for the training data in mini batch
    mode and then backpropagates to calculate the relevant gradients. Finally, the
    update step is done for all the weights.

    The method also includes the implementation of Adam Optimizer which performs 
    adaptive learning with momentum update.

    params:
    @alpha: learning rate
    @num_iterations: maximum number of iterations to perform  
*/
void NeuralNetwork::runGradientDescent(double alpha, int num_iterations)
{
    int iteration = 0;
    double prev_error = 1000000000, curr_error;
    while(iteration < num_iterations){
        forwardPropagation();
        backpropagation();

        for(int k=0; k<K; k++){
            for(int j=0; j<=M; j++){
                temp_kj[k][j] = weights_kj[k][j] - alpha * mt_kj_corrected[k][j] / (sqrt(vt_kj_corrected[k][j]) + epsilon);
            }
        }
        for(int j=0; j<M; j++){
            for(int i=0; i<=D; i++){
                temp_ji[j][i] = weights_ji[j][i] - alpha * mt_ji_corrected[j][i] / (sqrt(vt_ji_corrected[j][i]) + epsilon);
            }
        }

        curr_error = calculateError();
        if(curr_error > prev_error) break;
        prev_error = curr_error;

        for(int k=0; k<K; k++){
            for(int j=0; j<=M; j++){
                weights_kj[k][j] = temp_kj[k][j];
            }
        }
        for(int j=0; j<M; j++){
            for(int i=0; i<=D; i++){
                weights_ji[j][i] = temp_ji[j][i];
            }
        }
        iteration++;
        t++;
    }
    cout << "Iterations: " << iteration << endl;
}

/* Method to predict the output for a given set of test data and calculate
    the accuracy over the data.

    params:
    @data: test data of handwritten digits

    returns:
    @accuracy: the calculated accuracy
*/
double NeuralNetwork::predict(vector<pair<vector<double>, int> > data)
{
    int sz = data.size(), n = 0, ct1 = 0, ct2 = 0;
    vector<vector<double> > target_values;

    vector<double> input(D+1);

    for(int i=0; i<sz; i++){
        vector<double> row;
        for(int j=0; j<K; j++) row.push_back(0);
        target_values.push_back(row);
        target_values[i][data[i].second] = 1;
    }

    while(n < sz){
        for(int i=0; i<D; i++) input[i] = data[n].first[i];
        input[D] = 1;

        for(int j=0; j<M; j++){
            A_j[j] = 0;
            for(int i=0; i<=D; i++){
                A_j[j] += weights_ji[j][i] * input[i];
            }
            Z_j[0][j] = sigmoid(A_j[j]);
        }

        Z_j[0][M] = 1;
        for(int k=0; k<K; k++){
            A_k[k] = 0;
            for(int j=0; j<=M; j++){
                A_k[k] += weights_kj[k][j] * Z_j[0][j];
            }
            Y_k[k] = sigmoid(A_k[k]);
        }

        int predicted = 0, true_;
        double max = -1;
        for(int k=0; k<K; k++){
            if(Y_k[k] > max){
                max = Y_k[k];
                predicted = k;
            }
            if(target_values[n][k] == 1) true_ = k;
        }
        if(true_ == predicted) ct1++;
        else ct2++;
        n++;
    }
    double accuracy = (double)ct1/(ct1+ct2);
    return accuracy;
}