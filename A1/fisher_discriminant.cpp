#include <bits/stdc++.h>
#include "data_extraction.h"

using namespace std;

class fisher_discriminant
{
	public:
		vector<vector<float> > data;
		vector<float> miu1,miu2;
		fisher_discriminant(string file){
			extract_data training_data(file);		
			data = training_data.file_open(',');
		}
		
		void mean_calculation(){
			int pos_example=0,neg_example=0;
			int instances = 4;		//No of Instances in Training data
			vector<float> pos_EgSUM_xi(instances,0);
			vector<float> neg_EgSUM_xi(instances,0);
			int data_sz = data.size();
			for(int i=0;i<data_sz-1;i++){
				if(data[i][data[i].size()-1]==0){
					neg_example++;
					for(int j=0;j<neg_EgSUM_xi.size();j++){
						neg_EgSUM_xi[j]+=data[i][j];
					}
				}
				if(data[i][data[i].size()-1]==1){
					pos_example++;
					for(int j=0;j<pos_EgSUM_xi.size();j++){
						pos_EgSUM_xi[j]+=data[i][j];
					}
				}
			}
			for(int i=0;i<pos_EgSUM_xi.size();i++){
				miu1.push_back((pos_EgSUM_xi[i]/pos_example));
				miu2.push_back((neg_EgSUM_xi[i]/neg_example));
				//cout<<miu1[i]<<" "<<miu2[i]<<endl;
			}
		}	
		float sw_calculation(){
			this->mean_calculation();
			int data_sz = data.size();
			float sw=0;
			for(int i=0;i<data_sz-1;i++){
				if(data[i][data[i].size()-1]==0){
					for(int j=0;j<miu2.size();j++){
						sw+=(data[i][j]-miu2[j])*(data[i][j]-miu2[j]);	
					}					
				}
				if(data[i][data[i].size()-1]==1){
					for(int j=0;j<miu1.size();j++){
						sw+=(data[i][j]-miu1[j])*(data[i][j]-miu1[j]);	
					}
				}
			}
			return sw;
			//cout<<sw<<endl;
		}
		vector<float> w_calculation(){
			float sw = this->sw_calculation();
			std::vector<float> w;
			for(int i=0;i<miu1.size();i++){
				w.push_back(((miu2[i]-miu1[i])/sw));
			}
			/*for(int i=0;i<w.size();i++){
				cout<<w[i]<<" ";
			}*/
			return w;
		}
};