#include <bits/stdc++.h>
#include "./../A1/code/data_extraction.h"
using namespace std;
class NeuralNets
{
	private:
		vector<vector<double> > training_data, test_data, validation_data;		
		int hidden_layer_nodes,input_layer_nodes,output_layer_nodes,training_data_len;
		vector<vector<double> > weights_ji,weights_kj;
		vector<vector<double> > Z_i,A_k,A_j,Z_j,Y_k,derivative_wrt_Wkj,derivative_wrt_Wji;
		vector<vector<int> > T_k;	
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
			this->get_whole_data();
			this->generate_tk();
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
		void generate_tk(){
			std::vector<int> row;
			for(int j=0;j<=9;j++){
				T_k.push_back(row);
			}
			for(int i=0;i<training_data_len;i++){
				int x = training_data[i][64];
				for(int j=0;j<=9;j++){
					if(x==j){
						T_k[j].push_back(1);
					}
					else{
						T_k[j].push_back(0);
					}
				}
			}
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
		void Back_propogation(){
			vector<vector<double> >yk_minus_tk;
			for(int i=0;i<T_k.size();i++){
				vector<double> row;
				for(int j=0;j<T_k[i].size();j++){
					row.push_back(A_k[i][j]-T_k[i][j]);
				}
				yk_minus_tk.push_back(row);
			}
			for(int i=0;i<Z_j.size();i++){
				vector<double> row;
				for(int j=0;j<T_k.size();j++){
					double a=0;
					for(int k=0;k<T_k.size();k++){
						a+=Z_j[i][k]*(yk_minus_tk[j][k]);
					}
					row.push_back(a);
				}
				derivative_wrt_Wkj.push_back(row);
			}

			/*for(int i=0;i<derivative_wrt_Wkj.size();i++){
				for(int j=0;j<derivative_wrt_Wkj[i].size();j++){
					cout<<derivative_wrt_Wkj[i][j]<<" ";
				}
				cout<<endl;
			}*/
			
			vector<vector<double> > delta_j;
			vector<vector<double> > wk_yk_minus_tk;
			for(int i=0;i<hidden_layer_nodes;i++){
				vector<double> row;
				for(int j=0;j<yk_minus_tk[0].size();j++){
					double a=0;
					for(int k=0;k<output_layer_nodes;k++){
						a+=weights_kj[k][i]*yk_minus_tk[k][j];
					}
					row.push_back(a);
				}
				wk_yk_minus_tk.push_back(row);
			}
			
			vector<vector<double> > derivative_H_j;
			for(int k=0;k<Z_j.size();k++){
				vector<double> row;
				for(int i=0;i<Z_j.size();i++){
					double a=0;
					for(int j=0;j<Z_j[i].size();j++){
						a+=Z_j[k][j]*(1-Z_j[i][j]);
					}
					row.push_back(a);
				}
				derivative_H_j.push_back(row);
			}

			for(int i=0;i<derivative_H_j.size();i++){
				vector<double> row;
				for(int j=0;j<wk_yk_minus_tk[0].size();j++){
					double a=0;
					for(int k=0;k<wk_yk_minus_tk.size();k++){
						a+= derivative_H_j[i][j]*wk_yk_minus_tk[k][j];
					}
					row.push_back(a);
				}
				delta_j.push_back(row);
			}
			
			//Multiply Z_i (65*3100) matrix and delta_j (5*3100) matrix
			/*for(int i=0;i<input_layer_nodes;i++){
				for(int j=0;j<)
			}*/
		}
	
};
int main(){
	NeuralNets nn(5);
	nn.farward_prop_NN();
	nn.Back_propogation();
	//cout<<nn.training_data[0][0];	
	return 0;
}