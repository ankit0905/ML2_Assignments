#include <bits/stdc++.h>
using namespace std;

#define PI  3.14159265358
#define N 4

#include "helper.h"
#include "data_extraction.h"
#include "fisher_discriminant.cpp"
#include "prob_gen.cpp"
#include "logistic_regression.cpp"

int main(){
	string train_file,test_file;
	
	//change filename giving path to test or train in csv format 
	train_file = "../data/train.txt";
	test_file = "../data/test.txt";
	
	extract_data train_obj(train_file);
	extract_data test_obj(test_file);
	
	vector<vector<double> > training_data, test_data;
	training_data = train_obj.file_open(',');
	test_data = test_obj.file_open(',');
	
	Fisher_discriminant fd(training_data);
	fd.precision_recall(test_data);
	fd.printOutput();

	ProbGenrModel pgm(training_data);
	pgm.predict_2(test_data);
	pgm.printOutput();

	LogisticRegression lr(training_data);
	lr.predict(test_data);
	lr.printOutput();

	return 0;
}