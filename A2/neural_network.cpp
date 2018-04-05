class NeuralNetwork
{
    private:
        vector<pair<vector<double>, int> > training_data, test_data, validation_data;
        vector<vector<double> > weights_ji, weights_kj, T_k, d_E_wkj, d_E_wji, prediction;
        vector<double> X_i, A_j, A_k, Z_j, Y_k, delta_k, delta_j;
        int D, N, M, K, index = 0;
        const int batch_size = 100;

    public:
        NeuralNetwork(vector<pair<vector<double>, int> > train_data, int no_hidden_units);
        double sigmoid(double value);
        void generateTargetValues();
        void initializeWeights();
        void forwardPropagation();
        void backpropagation();
        void runGradientDescent(double alpha, int num_iterations);
        void predict(vector<pair<vector<double>, int> > train_data);
};

NeuralNetwork::NeuralNetwork(vector<pair<vector<double>, int> > train_data, int no_hidden_units=5)
{
    training_data = train_data;
    
    D = train_data[0].first.size();
    N = train_data.size();
    M = no_hidden_units;
    K = 10;

    cout << D << " " << N << " " << M << " " << K << endl;

    for(int j=0; j<M; j++){
        A_j.push_back(0);
        Z_j.push_back(0);
        delta_j.push_back(0);
    }
    Z_j.push_back(0);
    for(int j=0; j<=D; j++) X_i.push_back(0);
    for(int k=0; k<K; k++){
        Y_k.push_back(0);
        A_k.push_back(0);
        delta_k.push_back(0);
        vector<double> row;
        for(int j=0; j<=M; j++) row.push_back(0);
        d_E_wkj.push_back(row);
    }
    for(int j=0; j<M; j++){
        vector<double> row;
        for(int i=0; i<=D; i++) row.push_back(0);
        d_E_wji.push_back(row);
    }
    
    initializeWeights();
    generateTargetValues();
}

double NeuralNetwork::sigmoid(double value)
{
    return 1.0/(1.0+exp(-value));
}

void NeuralNetwork::generateTargetValues()
{
    for(int i=0; i<N; i++){
        vector<double> row;
        for(int j=0; j<K; j++) row.push_back(0);
        T_k.push_back(row);
        T_k[i][training_data[i].second] = 1;
    }
}

void NeuralNetwork::initializeWeights()
{
    for(int j=0; j<M; j++){
        vector<double> row;
        for(int i=0; i<=D; i++) row.push_back(-1);
        weights_ji.push_back(row);
    }
    for(int k=0; k<K; k++){
        vector<double> row;
        for(int j=0; j<=M; j++) row.push_back(-1);
        weights_kj.push_back(row);
    }
}

void NeuralNetwork::forwardPropagation()
{
    index = index % N;
    
    int counter = batch_size;
    double error;
    while(counter--){
        for(int i=0; i<D; i++) X_i[i] = training_data[index].first[i];
        X_i[D] = 1;

        // cout << "EXAMPLE: ";
        // for(int i=0; i<D; i++) cout << X_i[i] << " ";
        // cout << endl << endl;

        for(int j=0; j<M; j++){
            A_j[j] = 0;
            for(int i=0; i<=D; i++){
                A_j[j] += weights_ji[j][i]*X_i[i];
            }
            Z_j[j] = sigmoid(A_j[j]);
        }

        Z_j[M] = 1;
        for(int k=0; k<K; k++){
            A_k[k] = delta_k[k] = 0;
            for(int j=0; j<=M; j++){
                A_k[k] += weights_kj[k][j]*Z_j[j];
            }
            Y_k[k] = sigmoid(A_k[k]);
            delta_k[k] = (Y_k[k] - T_k[index][k]);
        }
        // cout << "OUTPUT: ";
        // for(int k=0; k<K; k++) cout << Y_k[k] << " ";
        // cout << endl << endl;
        index = (index + 1)%N;
    }
    for(int k=0; k<K; k++) delta_k[k] /= batch_size;
}

void NeuralNetwork::backpropagation()
{
    for(int j=0; j<M; j++){
        double sum = 0;
        for(int k=0; k<K; k++){
            sum += weights_kj[k][j] * delta_k[k];
        }
        delta_j[j] = sum * Z_j[j] * (1 - Z_j[j]);
    }

    for(int k=0; k<K; k++){
        for(int j=0; j<=M; j++){
            d_E_wkj[k][j] = delta_k[k] * Z_j[j];
        }
    }

    for(int j=0; j<M; j++){
        for(int i=0; i<=D; i++){
            d_E_wji[j][i] = delta_j[j] * X_i[i];
        }
    }
}

void NeuralNetwork::runGradientDescent(double alpha, int num_iterations)
{
    int iteration = 0;
    //cout << num_iterations << endl;
    while(iteration < num_iterations){
        //cout << endl << "============= ITERATION ============= " << endl;
        forwardPropagation();
        backpropagation();
        for(int k=0; k<K; k++){
            for(int j=0; j<=M; j++){
                weights_kj[k][j] = weights_kj[k][j] - alpha*d_E_wkj[k][j];
                cout << weights_kj[k][j] << " ";
            }
            cout << endl;
        }
        cout << endl << endl;
        for(int j=0; j<M; j++){
            for(int i=0; i<=D; i++){
                weights_ji[j][i] = weights_ji[j][i] - alpha*d_E_wji[j][i];
            }
        }
        iteration++;
    }
}

void NeuralNetwork::predict(vector<pair<vector<double>, int> > data)
{
    int sz = data.size(), n = 0;
    vector<vector<double> > target_values;
    
    for(int i=0; i<sz; i++){
        vector<double> row;
        for(int j=0; j<K; j++) row.push_back(0);
        target_values.push_back(row);
        target_values[i][data[i].second] = 1;
    }

    while(n < 10){
        for(int i=0; i<D; i++) X_i[i] = data[n].first[i];
        X_i[D] = 1;

        for(int j=0; j<M; j++){
            A_j[j] = 0;
            for(int i=0; i<=D; i++){
                A_j[j] += weights_ji[j][i]*X_i[i];
            }
            Z_j[j] = sigmoid(A_j[j]);
        }
        
        Z_j[M] = 1;
        for(int k=0; k<K; k++){
            A_k[k] = delta_k[k] = 0;
            for(int j=0; j<=M; j++){
                A_k[j] += weights_kj[k][j]*Z_j[j];
            }
            Y_k[k] = sigmoid(A_k[k]);
        }

        for(int k=0; k<K; k++) cout << Y_k[k] << " ";
        cout << endl;
        for(int k=0; k<K; k++) cout << target_values[n][k] << " ";
        cout << endl << endl;
        n++;
    }
}