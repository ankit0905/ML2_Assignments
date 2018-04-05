#include <bits/stdc++.h>
using namespace std;

#include "helper.h"
#include "neural_network.cpp"

int main()
{
    vector<pair<vector<double>, int> > train_data = loadDataset("data/train.csv");
    vector<pair<vector<double>, int> > validation_data = loadDataset("data/validation.txt");
    vector<pair<vector<double>, int> > test_data = loadDataset("data/test.txt");
    for(int i=6; i<=10; i++){
        double accuracy = 0;
        cout << "HIDDEN UNITS: " << i << endl;
        NeuralNetwork nn(validation_data, train_data, i);
        nn.runGradientDescent(0.01,3000);
        accuracy = nn.predict(test_data);
        cout << "Accuracy: " << accuracy << endl << endl;
    }
    return 0;
}