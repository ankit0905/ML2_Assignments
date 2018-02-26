#include <bits/stdc++.h>
#include "data_extraction.h"

using namespace std;

class fisher_discriminant
{
	public:
		vector<vector<double> > data;
		vector<double> miu1,miu2;
		int data_sz;
		fisher_discriminant(string file){
			extract_data training_data(file);		
			data = training_data.file_open(',');
			data_sz=data.size();
		}
		
		void mean_calculation(){
			int pos_example=0,neg_example=0;
			int instances = 4;		//No of Instances in Training data
			vector<double> pos_EgSUM_xi(instances,0);
			vector<double> neg_EgSUM_xi(instances,0);
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
		double sw_calculation(){
			this->mean_calculation();
			double sw=0;
			for(int i=0;i<data_sz-1;i++){
				if(data[i][data[i].size()-1]==0){
					for(int j=0;j<miu2.size();j++){
						sw+=((data[i][j]-miu2[j])*(data[i][j]-miu2[j]));	
					}					
				}
				if(data[i][data[i].size()-1]==1){
					for(int j=0;j<miu1.size();j++){
						sw+=((data[i][j]-miu1[j])*(data[i][j]-miu1[j]));	
					}
				}
			}
			//cout<<sw<<endl;
			return sw;
		}
		vector<double> w_calculation(){
			double sw = this->sw_calculation();
			std::vector<double> w;
			for(int i=0;i<miu1.size();i++){
				w.push_back(((miu2[i]-miu1[i])/sw));
			}
			return w;
		}
		double threshold_calculation(){
			vector<double> w=this->w_calculation();
			vector<double> pt1D;
			for(int i=0;i<data_sz;i++){
				double y_x=0;
				for(int j=0;j<data[i].size();j++){
					y_x+=(data[i][j]*w[j]);
				}
				pt1D.push_back(y_x);
			}
			sort(pt1D.begin(),pt1D.end(), std::greater<double>());
			double min_etpy = DBL_MAX; 
			double threshold;
			int p,nn;
			for(int i=0;i<pt1D.size()-1;i++){
				double f = (pt1D[i]+pt1D[i+1])/2;
				int pos=0,neg=0;
				for(int j=0;j<pt1D.size();j++){
					if(pt1D[j]<=f){
						neg++;
					}
					else{
						pos++;
					}
				}
				double total = pos+neg;
				double entropy = -1*(((pos/total) * log(((pos/total)))) + (((neg/total))*log((neg/total))));
				if(entropy<min_etpy){
					min_etpy=entropy;
					threshold=f;
					p = pos;
					nn = neg;
				}
			}
			cout<<p<<" "<<nn<<endl;
			return threshold;
		
		}
};