#include <iostream>
#include <algorithm>
#include <vector>
#include <fstream>

using namespace std;

class extract_data{
	public:
		string filename;
		extract_data(string file){
			filename = file;
		}
		std::vector<std::vector<double> > file_open(char delimiter){
			ifstream inFile;
			inFile.open(filename);
			if(!inFile){
				cout<<"Error In Opening File";
				exit(1);
			}
			string occurence;
			int lineno = 0;
			vector < vector< double > > data(3100);
			while(inFile >> occurence){
				int delim_occur = occurence.find(delimiter);
				string occur;
				occur = occurence;
				while(delim_occur!=-1){
					data[lineno].push_back(stof(occur.substr(0,delim_occur)));
					occur = occur.substr(delim_occur+1,occur.length());
					delim_occur = occur.find(delimiter);
				}
				data[lineno].push_back(stof(occur.substr(0,delim_occur)));
				occur = occur.substr(delim_occur+1,occur.length());
				delim_occur = occur.find(delimiter);
				/*for(int i=0;i<data[lineno].size();i++){
					cout<<data[lineno][i]<<",";
				}
				cout<<endl;*/
				lineno++;
			}
			return data;
			inFile.close(); 	
		}
};