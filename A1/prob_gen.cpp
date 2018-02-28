#include "helper.h"

class ProbGenrModel
{
	private:
		vector<vector<double> > data;
		int rows, cols, count_c0 = 0, count_c1 = 0;
		int true_negative = 0, true_positive = 0, false_negative = 0, false_positive = 0;
		vector<double> mean_c0, mean_c1, sum_c0, sum_c1, weights;
		double covariance_matrix[4][4], bias;
		bool error = false;
	
	public:
		ProbGenrModel(vector<vector<double> > data);
		void mean_calculation();
		void covariance_calculation();
		void computeWeights();
		void predict(vector<vector<double> > test_data);
		void printOutput();
};

ProbGenrModel::ProbGenrModel(vector<vector<double> > data)
{
	this -> data = data;
	rows = data.size();
	cols = 5;
	mean_calculation();
	covariance_calculation();
	computeWeights();
}

void ProbGenrModel::mean_calculation()
{
	int class_;
	for(int i=0; i<cols-1; i++){
		sum_c0.push_back(0.0);
		sum_c1.push_back(0.0);
	}
	for(int i=0; i<rows; i++){
		if(data[i][cols-1] == 0) class_ = 0;
		else class_ = 1;
		for(int j=0; j<cols-1; j++){
			if(class_ == 0) sum_c0[j] += data[i][j];
			else sum_c1[j] += data[i][j];
		}
		if(class_ == 0) count_c0++;
		else count_c1++;
	}
	for(int i=0; i<cols-1; i++){
		mean_c0.push_back(sum_c0[i]/count_c0);
		mean_c1.push_back(sum_c1[i]/count_c1);
	}
}

void ProbGenrModel::covariance_calculation()
{
	double mean_xi, mean_xj, sum;
	for(int i=0; i<cols-1; i++){
		for(int j=i; j<cols-1; j++){
			mean_xi = (mean_c0[i]*count_c0 + mean_c1[i]*count_c1)/rows;
			mean_xj = (mean_c0[j]*count_c0 + mean_c1[j]*count_c1)/rows;
			sum = 0.0;
			for(int k=0; k<rows; k++){
				sum = sum + (data[k][i]-mean_xi)*(data[k][j]-mean_xj);
			}
			sum = sum/rows;
			covariance_matrix[i][j] = covariance_matrix[j][i] = sum;
		}
	}
}

void ProbGenrModel::computeWeights()
{
	vector<vector<double> > sigma_inverse(4, vector<double>(4));
	double sigma_inverse_[4][4];
	double sum;
	if(!inverse(covariance_matrix, sigma_inverse_)){
		error = true;
		return;
	}
	for(int i=0; i<cols-1; i++){
		for(int j=0; j<cols-1; j++){
			sigma_inverse[i][j] = sigma_inverse_[i][j];
		}
	}
	for(int i=0; i<cols-1; i++){
		sum = 0.0;
		for(int j=0; j<cols-1; j++){
			sum += sigma_inverse[i][j]*(mean_c0[j] - mean_c1[j]);
		}
		weights.push_back(sum);
	}

	bias = 0.0;
	vector<vector<double> > mean(cols-1, vector<double>(1));
	vector<vector<double> > mean_T(1, vector<double>(cols-1));

	for(int i=0; i<cols-1; i++) mean_T[0][i] = mean[i][0] = mean_c0[i];

	vector<vector<double> > res(1, vector<double>(1));
	vector<vector<double> > tmp(1, vector<double>(cols-1));

	multiply(1,cols-1,cols-1,cols-1,mean_T,sigma_inverse,tmp);
	multiply(1,cols-1,cols-1,1,tmp,mean,res);

	bias = -0.5*res[0][0];
	
	for(int i=0; i<cols-1; i++) mean_T[0][i] = mean[i][0] = mean_c1[i];
	multiply(1,cols-1,cols-1,cols-1,mean_T,sigma_inverse,tmp);
	multiply(1,cols-1,cols-1,1,tmp,mean,res);

	bias += 0.5*res[0][0];
	bias += log((double)count_c0/count_c1);
}

void ProbGenrModel::predict(vector<vector<double> > test_data)
{
	int size = test_data.size();
	double value;
	//cout << size << " " << test_data[0].size() << " " << weights.size() << endl;
	int predicted_class, true_class;
	for(int i=0; i<412; i++){
		//check the size of test_data
		value = 0.0;
		for(int j=0; j<cols-1; j++){
			value += weights[j]*test_data[i][j];
		}
		value = sigmoid(value + bias);
		//cout << i << " " << value << endl;
		if(value <= 0.5) predicted_class = 1;
		else predicted_class = 0;
		true_class = test_data[i][4];
		if(true_class == predicted_class){
			if(true_class == 0) true_negative++;
			else true_positive++;
		}
		else{
			if(true_class == 0) false_positive++;
			else false_negative++;
		}
	}
	
}

void ProbGenrModel::printOutput()
{
	double precision = (double)true_positive/(true_positive+false_positive);
	double recall = (double)true_positive/(true_positive+false_negative);
	cout << "PROBABILISTIC GENERATIVE MODEL" << endl;
	cout << "Precision: " << precision << endl;
	cout << "Recall: " << recall << endl << endl;

	cout << "CONFUSION MATRIX" << endl;
	cout << "\t  Predicted = 0\t  Predicted=1" << endl;
	cout << "Actual=0: " << true_negative << "\t\t  " << false_positive << endl;
	cout << "Actual=1: " << false_negative << "\t\t  " << true_positive << endl;
}