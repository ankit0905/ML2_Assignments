#include <bits/stdc++.h>
//#include "data_extraction.h"
#include "fisher_discriminant.cpp"

using namespace std;

int main(){
	string file;
	//change filename giving path to test or train in csv format 
	file = "/home/tex/Documents/ML/ML2_Assignments/A1/train.txt";
	/*extract_data data(file);
	vector<std::vector<float> >v;
	v = data.file_open(',');*/

	fisher_discriminant fd(file);
	
	vector<float> w =fd.w_calculation();
	return 0;
}