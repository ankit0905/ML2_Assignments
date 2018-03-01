#include <bits/stdc++.h>
using namespace std;

#include "data_extraction.h"
#include "fisher_discriminant.cpp"
#include "prob_gen.cpp"

int main(){
	string train_file,test_file;
	
	//change filename giving path to test or train in csv format 
	train_file = "data/train.txt";
	test_file = "data/test.txt";
	
	extract_data train_obj(train_file);
	extract_data test_obj(test_file);
	
	vector<vector<double> > training_data, test_data;
	training_data = train_obj.file_open(',');
	test_data = test_obj.file_open(',');
	
	//fisher_discriminant fd(train_file);
	//fd.precision_recall(test_file);

	ProbGenrModel pgm(training_data);
	pgm.predict(test_data);
	pgm.printOutput();

	return 0;
}