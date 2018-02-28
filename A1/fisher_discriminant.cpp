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
		 
		    // Looping for each element of the matrix
		    for (int row = 0; row < n; row++)
		    {
		        for (int col = 0; col < n; col++)
		        {
		            //  Copying into temporary matrix only those element
		            //  which are not in given row and column
		            if (row != p && col != q)
		            {
		                temp[i][j++] = A[row][col];
		 
		                // Row is filled, so increase row index and
		                // reset col index
		                if (j == n - 1)
		                {
		                    j = 0;
		                    i++;
		                }
		            }
		        }
		    }
		}
		 
		/* Recursive function for finding determinant of matrix.
		   n is current dimension of A[][]. */
		double determinant(double A[N][N], int n)
		{
		    double D = 0; // Initialize result
		 
		    //  Base case : if matrix contains single element
		    if (n == 1)
		        return A[0][0];
		 
		    double temp[N][N]; // To store cofactors
		 
		    double sign = 1;  // To store sign multiplier
		 
		     // Iterate for each element of first row
		    for (int f = 0; f < n; f++)
		    {
		        // Getting Cofactor of A[0][f]
		        this->getCofactor(A, temp, 0, f, n);
		        D += sign * A[0][f] * this->determinant(temp, n - 1);
		 
		        // terms are to be added with alternate sign
		        sign = -sign;
		    }
		 
		    return D;
		}
		 
		// Function to get adjoint of A[N][N] in adj[N][N].
		void adjoint(double A[N][N],double adj[N][N])
		{
		    if (N == 1)
		    {
		        adj[0][0] = 1;
		        return;
		    }
		 
		    // temp is used to store cofactors of A[][]
		    double sign = 1, temp[N][N];
		 
		    for (int i=0; i<N; i++)
		    {
		        for (int j=0; j<N; j++)
		        {
		            // Get cofactor of A[i][j]
		            this->getCofactor(A, temp, i, j, N);
		 
		            // sign of adj[j][i] positive if sum of row
		            // and column indexes is even.
		            sign = ((i+j)%2==0)? 1: -1;
		 
		            // Interchanging rows and columns to get the
		            // transpose of the cofactor matrix
		            adj[j][i] = (sign)*(this->determinant(temp, N-1));
		        }
		    }
		}
		 
		// Function to calculate and store inverse, returns false if
		// matrix is singular
		bool inverse(double A[N][N], double inverse[N][N])
		{
		    // Find determinant of A[][]
		    double det = this->determinant(A, N);
		    if (det == 0)
		    {
		        cout << "Singular matrix, can't find its inverse";
		        return false;
		    }
		 
		    // Find adjoint
		    double adj[N][N];
		    this->adjoint(A, adj);
		 
		    // Find Inverse using formula "inverse(A) = adj(A)/det(A)"
		    for (int i=0; i<N; i++)
		        for (int j=0; j<N; j++)
		            inverse[i][j] = adj[i][j]/det;
		 
		    return true;
		}
		 
		// Generic function to display the matrix.  We use it to display
		// both adjoin and inverse. adjoin is integer matrix and inverse
		// is a float.
		
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
		vector<vector<double> > sw_calculation(){
			this->mean_calculation();
			vector<vector<double> > sw {{0,0,0,0},
										{0,0,0,0},
										{0,0,0,0},
										{0,0,0,0}};

			for(int i=0;i<data_sz-1;i++){
				if(data[i][data[i].size()-1]==0){
					for(int j=0;j<miu1.size();j++){
						for(int k=0;k<miu1.size();k++){
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
			/*for(int i=0;i<sw.size();i++){
				for(int j=0;j<sw[i].size();j++){
					cout<<sw[i][j]<<" ";
				}
				cout<<endl;
			}*/
			return sw;
		}
		vector<double> w_calculation(){
			vector<vector<double> > sw = this->sw_calculation();
			std::vector<double> w_;
			double sw_inverse[4][4],Mat[4][4];
			for(int i=0;i<4;i++){
				for(int j=0;j<4;j++){
					//sw_inverse[i][j]=sw[i][j];
					Mat[i][j]=sw[i][j];
				}
			}
			this->inverse(Mat,sw_inverse);
			for(int i=0;i<miu1.size();i++){
				double val=0;
				for(int j=0;j<miu1.size();j++){
					val+=((miu2[j]-miu1[j]))*sw_inverse[i][j];
				}
				w_.push_back(val);
			}
			/*for(int i=0;i<w.size();i++){
				cout<<w[i]<<" ";
			}*/
			return w_;
		}
		double threshold_calculation(){
			w=this->w_calculation();
			vector<double> pt1D;
			for(int i=0;i<data_sz;i++){
				double y_x=0;
				for(int j=0;j<data[i].size();j++){
					y_x+=(data[i][j]*w[j]);
				}
				pt1D.push_back(y_x);
			}
			sort(pt1D.begin(),pt1D.end(),std::greater<double>());
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
			return threshold;		
		}
		void precision_recall(string file){
			double threshold = this->threshold_calculation();
			extract_data training_data(file);		
			vector<vector<double> >  test_data = training_data.file_open(',');
			cout<<w.size()<<endl;
			int true_pos=0,false_pos=0,true_neg=0,false_neg=0;
			for(int i=0;i<test_data.size();i++){
				double wx=0;
				for(int j=0;j<w.size();j++){
					wx+=w[j]*test_data[i][j];
				}
				if(wx>=threshold && test_data[i][5]==1){
					true_pos++;
				}
				else if(wx>=threshold && test_data[i][5]==0){
					false_pos++;
				}
				else if(wx<threshold && test_data[i][5]==0){
					true_neg++;
				}
				else if(wx<threshold && test_data[i][5]==1){
					false_neg++;
				}
			}
			double precision = (true_pos)/(true_pos+false_pos); 
			double recall = (true_pos)/(true_pos+false_neg);
			cout<<precision<<" "<<recall<<endl;
		}
};