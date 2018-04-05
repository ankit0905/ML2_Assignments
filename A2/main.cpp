#include <bits/stdc++.h>
using namespace std;

#include "helper.h"
#include "neural_network.cpp"

int main()
{
    vector<pair<vector<double>, int> > train_data = loadDataset("data/train.csv");
    vector<pair<vector<double>, int> > validation_data = loadDataset("data/validation.txt");
    vector<pair<vector<double>, int> > test_data = loadDataset("data/test.txt");
    NeuralNetwork nn(train_data);
    nn.runGradientDescent(0.8,10);
    nn.predict(validation_data);
    return 0;
}