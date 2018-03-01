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
		void predict_2(vector<vector<double> > test_data);
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
		mean_c0.push_back(sum_c0[i]/(double)count_c0);
		mean_c1.push_back(sum_c1[i]/(double)count_c1);
	}
}

void ProbGenrModel::covariance_calculation()
{
	double sum1, sum2;
	for(int i=0; i<cols-1; i++){
		for(int j=i; j<cols-1; j++){
			sum1 = sum2 = 0.0;
			for(int k=0; k<rows; k++){
				if(data[k][4] == 0)
					sum1 = sum1 + (data[k][i]-mean_c0[i])*(data[k][j]-mean_c0[j]);
				else
					sum2 = sum2 + (data[k][i]-mean_c1[i])*(data[k][j]-mean_c1[j]);
			}
			covariance_matrix[i][j] = covariance_matrix[j][i] = (sum1+sum2)/(double)rows;
		}
	}
}

void ProbGenrModel::predict_2(vector<vector<double> > test_data)
{
	double pX_C0, pX_C1;
	double pC0 = (double)count_c0/rows, pC1 = (double)count_c1/rows;
	vector<vector<double> > mean_diff(cols-1, vector<double>(1)), mean_diff_T(1, vector<double>(cols-1));
	vector<vector<double> > tmp(1, vector<double>(cols-1));//, cov(cols-1, vector<double>(cols-1));
	vector<vector<double> > res(1, vector<double>(1));

	vector<vector<double> > sigma_inverse(cols-1, vector<double>(cols-1));
	double sigma_inverse_[4][4];

	int predicted_class, true_class;

	if(!inverse(covariance_matrix, sigma_inverse_)){
		error = true;
		return;
	}	

	for(int i=0; i<cols-1; i++){
		for(int j=0; j<cols-1; j++){
			sigma_inverse[i][j] = sigma_inverse_[i][j];
		}
	}

	for(int i=0; i<412; i++){
		for(int j=0; j<cols-1; j++){
			mean_diff[j][0] = mean_diff_T[0][j] = test_data[i][j] - mean_c0[j];
		}
		multiply(1,cols-1,cols-1,cols-1,mean_diff_T,sigma_inverse,tmp);
		multiply(1,cols-1,cols-1,1,tmp,mean_diff,res);

		pX_C0 = log(pC0) - 0.5*res[0][0];

		for(int j=0; j<cols-1; j++){
			mean_diff[j][0] = (test_data[i][j] - mean_c1[j]);
			mean_diff_T[0][j] = (test_data[i][j] - mean_c1[j]);
		}

		multiply(1,cols-1,cols-1,cols-1,mean_diff_T,sigma_inverse,tmp);
		multiply(1,cols-1,cols-1,1,tmp,mean_diff,res);
	
		pX_C1 = log(pC1) - 0.5*res[0][0];
		true_class = test_data[i][4];

		if(pX_C0 >= pX_C1) predicted_class = 0;
		else predicted_class = 1;

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
	int predicted_class, true_class;
	for(int i=0; i<412; i++){
		value = 0.0;
		for(int j=0; j<cols-1; j++){
			value += weights[j]*test_data[i][j];
		}
		value = sigmoid(value + bias);
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
	double accuracy = (double)(true_positive+true_negative)/(true_positive+
							true_negative+false_negative+false_positive);
	double precision = (double)true_positive/(true_positive+false_positive);
	double recall = (double)true_positive/(true_positive+false_negative);
	cout << "TASK#2: PROBABILISTIC GENERATIVE MODEL" << endl;
	cout << "    Accuracy: " << accuracy << endl;
	cout << "    Precision: " << precision << endl;
	cout << "    Recall: " << recall << endl << endl;

	cout << "  CONFUSION MATRIX" << endl;
	cout << "\t      Predicted = 0\t  Predicted=1" << endl;
	cout << "    Actual=0: " << true_negative << "\t\t  " << false_positive << endl;
	cout << "    Actual=1: " << false_negative << "\t\t\t " << true_positive << endl;
}