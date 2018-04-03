#include <bits/stdc++.h>
#include "./../A1/code/data_extraction.h"
using namespace std;
class NeuralNets
{
	private:
		vector<vector<double> > training_data, test_data, validation_data;		
		int hidden_layer_nodes,input_layer_nodes,output_layer_nodes,training_data_len;
		vector<vector<double> > weights_ji,weights_kj;
		vector<vector<double> > Z_i;
		vector<vector<double> > A_j;
		vector<vector<double> > Z_j;
		vector<vector<double> > A_k;	
	public:
		NeuralNets(int nodes){
			hidden_layer_nodes = nodes;
			input_layer_nodes = 64;
			output_layer_nodes = 10;
			training_data_len = 3100;
			//int random_no = (2*((float)rand()/RAND_MAX)) - 1 ;
			for(int i=0;i<hidden_layer_nodes;i++){
				vector<double> row;
				for(int j=0;j<65;j++){
					row.push_back((2*((float)rand()/RAND_MAX))-1);
				}
				weights_ji.push_back(row);
			}
			for(int i=0;i<output_layer_nodes;i++){
				vector<double> row;
				for(int j=0;j<10;j++){
					row.push_back((2*((float)rand()/RAND_MAX))-1);	
				}
				weights_kj.push_back(row);
			}
		}
		vector<vector<double> > get_data(string filename){
			extract_data obj(filename);
			return obj.file_open(',');
		}
		void get_whole_data(){
			string train_file,test_file,validation_file;
			train_file = "./train.csv";
			test_file = "./test.csv";
			validation_file = "./validation.csv";		
			this->training_data = get_data(train_file);
			this->test_data = get_data(test_file);
			this->validation_data = get_data(validation_file);
		}
		double sigmoid(double value){
    		return 1.0/(1.0+exp(-value));
		}
		void farward_prop_NN(){
			
			{
				vector<double> row;
				for(int i=0;i<training_data_len;i++){
					row.push_back(sigmoid(1));
				}
				Z_i.push_back(row);
			}

			for(int i=0;i<64;i++){
				vector<double> row;
				for (int j=0;j<training_data_len;j++){
					row.push_back(sigmoid(training_data[j][i]));
				}
				Z_i.push_back(row);	
			}
			
			
			for(int k=0;k<hidden_layer_nodes;k++){
				vector<double> row;
				for(int j=0;j<Z_i[0].size();j++){
					double a=0;
					for(int i=0;i<Z_i.size();i++){
						a+=Z_i[i][j]*weights_ji[k][i];
					}
					row.push_back(a);
				}
				A_j.push_back(row);				
			}
			
			for(int i=0;i<A_j.size();i++){
				vector<double> row;
				for(int j=0;j<A_j[i].size();j++){
					row.push_back(sigmoid(A_j[i][j]));
				}
				Z_j.push_back(row);
			}
						
			for(int i=0;i<output_layer_nodes;i++){
				vector<double> row;
				for(int j=0;j<Z_j[0].size();j++){
					double a=0;
					for(int k=0;k<Z_j.size();k++){
						a+=Z_j[k][j]*weights_kj[i][k];
					}
					row.push_back(a);
				}
				A_k.push_back(row);
			}
			cout<<"DONE farward_prop"<<endl;
		}
	
};
int main(){
	NeuralNets nn(5);
	nn.get_whole_data();
	nn.farward_prop_NN();
	//cout<<nn.training_data[0][0];	
	return 0;
}