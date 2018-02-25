#include <bits/stdc++.h>
#include "data_extraction.h"

using namespace std;

int main(){
	string file;
	//change filename giving path in your system
	file = "/home/tex/Documents/ML/ML2_Assignments/A1/train.txt";
	extract_data data(file);
	vector<std::vector<float> >v;
	v = data.file_open(',');
	return 0;
}