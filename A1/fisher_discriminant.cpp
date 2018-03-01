#include <bits/stdc++.h>
#include "data_extraction.h"
#define N 4

using namespace std;

class fisher_discriminant
{
	public:
		vector<vector<double> > data;
		vector<double> miu1,miu2;
		vector<double> w;
		int data_sz;
		double threshold;
		void getCofactor(double A[N][N], double temp[N][N], int p, int q, int n)
		{
		    int i = 0, j = 0;
		 
		    for (int row = 0; row < n; row++)
		    {
		        for (int col = 0; col < n; col++)
		        {
		            if (row != p && col != q)
		            {
		                temp[i][j++] = A[row][col];
		                if (j == n - 1)
		                {
		                    j = 0;
		                    i++;
		                }
		            }
		        }
		    }
		}
		double determinant(double A[N][N], int n)
		{
		    double D = 0;
		 
		    if (n == 1)
		        return A[0][0];
		 
		    double temp[N][N];
		 
		    double sign = 1;
		 
		    for (int f = 0; f < n; f++)
		    {
		        this->getCofactor(A, temp, 0, f, n);
		        D += sign * A[0][f] * this->determinant(temp, n - 1);
		        sign = -sign;
		    }
		 
		    return D;
		}
		
		void adjoint(double A[N][N],double adj[N][N])
		{
		    if (N == 1)
		    {
		        adj[0][0] = 1;
		        return;
		    }
		    double sign = 1, temp[N][N];
		 
		    for (int i=0; i<N; i++)
		    {
		        for (int j=0; j<N; j++)
		        {
		            this->getCofactor(A, temp, i, j, N);
		 	        sign = ((i+j)%2==0)? 1: -1;
		            adj[j][i] = (sign)*(this->determinant(temp, N-1));
		        }
		    }
		}
		bool inverse(double A[N][N], double inverse[N][N])
		{
		    // Find determinant of A[][]
		    double det = this->determinant(A, N);
		    if (det == 0)
		    {
		        cout << "Singular matrix, can't find its inverse";
		        return false;
		    }
		 	double adj[N][N];
		    this->adjoint(A, adj);
		 	for (int i=0; i<N; i++)
		        for (int j=0; j<N; j++)
		            inverse[i][j] = adj[i][j]/det;
		 
		    return true;
		}
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
			}
		}	
		vector<vector<double> > sw_calculation(){
			this->mean_calculation();
			vector<vector<double> > sw {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}};
			for(int i=0;i<data_sz;i++){
				if(data[i][data[i].size()-1]==0){
					for(int j=0;j<miu2.size();j++){
						for(int k=0;k<miu2.size();k++){
							sw[j][k]+=((data[i][j]-miu2[j])*(data[i][k]-miu2[k]));
						}
					}
				}
				if(data[i][data[i].size()-1]==1){
					for(int j=0;j<miu1.size();j++){
						for(int k=0;k<miu1.size();k++){
							sw[j][k]+=((data[i][j]-miu1[j])*(data[i][k]-miu1[k]));
						}
					}
				}
			}
			return sw;
		}
		vector<double> w_calculation(){
			vector<vector<double> > sw = this->sw_calculation();
			std::vector<double> w_;
			double sw_inverse[4][4],Mat[4][4];
			for(int i=0;i<4;i++){
				for(int j=0;j<4;j++){
					Mat[i][j]=sw[i][j];
				}
			}
			this->inverse(Mat,sw_inverse);
			for(int i=0;i<miu1.size();i++){
				double val=0;
				for(int j=0;j<miu1.size();j++){
					val+=((miu1[j]-miu2[j]))*sw_inverse[i][j];
				}
				w_.push_back(val);
			}
			return w_;
		}
		double threshold_calculation(){
			w=this->w_calculation();
			vector<pair<double,double> >pt1D;
			
			for(int i=0;i<data_sz;i++){
				double y_x=0;
				for(int j=0;j<w.size();j++){
					y_x+=(data[i][j]*w[j]);
				}
				pt1D.push_back(make_pair(y_x,data[i][4]));
			}
			sort(pt1D.begin(),pt1D.end());
			reverse(pt1D.begin(),pt1D.end());
			double min_etpy = DBL_MAX; 
			double threshold;
			for(int i=0;i<pt1D.size()-1;i++){
				double f = (pt1D[i].first+pt1D[i+1].first)/2;
				double pos0=0,neg0=0,pos1=0,neg1=0;
				for(int j=0;j<pt1D.size();j++){
					if(pt1D[j].first<f && pt1D[j].second==1){
						pos0++;
					}
					if(pt1D[j].first<f && pt1D[j].second==0){
						neg0++;
					}
					if(pt1D[j].first>=f && pt1D[j].second==0){
						neg1++;
					}
					if(pt1D[j].first>=f && pt1D[j].second==1){
						pos1++;
					}
				}
				double entropy = (-1*(((pos0/(pos0+neg0)) * (log(((pos0/(pos0+neg0))))/(log(2)))) + (((neg0/(pos0+neg0)))*(log((neg0/(pos0+neg0)))/(log(2))))))
								+(-1*(((pos1/(pos1+neg1)) * (log(((pos1/(pos1+neg1))))/(log(2)))) + (((neg1/(pos1+neg1)))*(log((neg1/(pos1+neg1)))/(log(2))))));
				
				if(entropy<min_etpy){
					min_etpy=entropy;
					threshold=f;
				}
			}
			return threshold;		
		}
		void precision_recall(string tstfile){
			double threshold = this->threshold_calculation();
			extract_data training_data(tstfile);		
			vector<vector<double> >  test_data = training_data.file_open(',');
			double true_pos=0,false_pos=0,true_neg=0,false_neg=0;
			cout<<"W_transpose is"<<endl;
			for(int i=0;i<w.size();i++){
				cout<<w[i]<<" ";
			}
			cout<<endl;
			for(int i=0;i<412;i++){
				double wx=0;
				for(int j=0;j<w.size();j++){
					wx+=w[j]*test_data[i][j];
				}
				int predicted_class,true_class;
				if(wx<threshold){
					predicted_class=0;
					//cout<<0<<endl;
				}
				else{
					predicted_class=1;
					//cout<<1<<endl;
				}
				true_class = test_data[i][4];
				if(true_class == predicted_class){
					if(true_class == 0) true_neg++;
					else true_pos++;
				}
				else{
					if(true_class == 0) false_pos++;
					else false_neg++;
				}
			}
			double precision = (true_pos)/(true_pos+false_pos); 
			double recall = (true_pos)/(true_pos+false_neg);
			cout<<"Precision=\t"<<precision<<"\nRecall   =\t"<<recall<<endl;
			//cout<<true_pos<<" "<<false_pos<<" "<<false_neg<<" "<<true_neg<<endl;
		}
};