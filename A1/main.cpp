#include <bits/stdc++.h>
//#include "data_extraction.h"
#include "fisher_discriminant.cpp"

using namespace std;

int main(){
	string file;
	//change filename giving path to test or train in csv format 
	file = "/home/tex/Documents/ML/ML2_Assignments/A1/train.txt";
	/*extract_data data(file);
	vector<std::vector<double> >v;
	v = data.file_open(',');*/

	fisher_discriminant fd(file);
	
	//vector<double> w =fd.w_calculation();
	fd.threshold_calculation();
	return 0;
}