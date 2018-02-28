#include <bits/stdc++.h>
//#include "data_extraction.h"
#include "fisher_discriminant.cpp"

using namespace std;

int main(){
	string train_file,test_file;
	//change filename giving path to test or train in csv format 
	train_file = "/home/tex/Documents/ML/ML2_Assignments/A1/train.txt";
	test_file = "/home/tex/Documents/ML/ML2_Assignments/A1/test.txt";
	/*extract_data data(file);
	vector<std::vector<double> >v;
	v = data.file_open(',');*/

	fisher_discriminant fd(train_file);
	fd.precision_recall(test_file);
	return 0;
}