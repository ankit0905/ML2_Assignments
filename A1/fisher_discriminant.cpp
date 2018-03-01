class Fisher_discriminant
{
	private:
		vector<vector<double> > data;
		vector<double> miu1,miu2;
		vector<double> w;
		int data_sz;
		double threshold;
		double true_positive=0,false_positive=0,true_negative=0,false_negative=0;

	public:
		Fisher_discriminant(vector<vector<double> > training_data);
		void mean_calculation();
		vector<vector<double> > sw_calculation();
		vector<double> w_calculation();
		double threshold_calculation();
		void precision_recall(vector<vector<double> >  test_data);
		void printOutput();
};

Fisher_discriminant::Fisher_discriminant(vector<vector<double> > training_data){		
	data = training_data;
	data_sz = data.size();
}

void Fisher_discriminant::mean_calculation(){
	int pos_example=0,neg_example=0;
	int instances = 4;		//No of Instances in Training data
	vector<double> pos_EgSUM_xi(instances,0);
	vector<double> neg_EgSUM_xi(instances,0);
	for(int i=0; i<data_sz-1; i++){
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
	for(int i=0; i<pos_EgSUM_xi.size(); i++){
		miu1.push_back((pos_EgSUM_xi[i]/pos_example));
		miu2.push_back((neg_EgSUM_xi[i]/neg_example));
	}
}	

vector<vector<double> > Fisher_discriminant::sw_calculation(){
	this->mean_calculation();
	vector<vector<double> > sw {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}};
	for(int i=0; i<data_sz; i++){
		if(data[i][data[i].size()-1]==0){
			for(int j=0; j<miu2.size(); j++){
				for(int k=0; k<miu2.size(); k++){
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

vector<double> Fisher_discriminant::w_calculation(){
	vector<vector<double> > sw = this->sw_calculation();
	std::vector<double> w_;
	double sw_inverse[4][4],Mat[4][4];
	for(int i=0;i<4;i++){
		for(int j=0;j<4;j++){
			Mat[i][j]=sw[i][j];
		}
	}
	inverse(Mat,sw_inverse);
	for(int i=0;i<miu1.size();i++){
		double val=0;
		for(int j=0;j<miu1.size();j++){
			val+=((miu1[j]-miu2[j]))*sw_inverse[i][j];
		}
		w_.push_back(val);
	}
	return w_;
}

double Fisher_discriminant::threshold_calculation(){
	w = this->w_calculation();
	vector<pair<double,double> >pt1D;
	
	for(int i=0; i<data_sz; i++){
		double y_x=0;
		for(int j=0; j<w.size(); j++){
			y_x+=(data[i][j]*w[j]);
		}
		pt1D.push_back(make_pair(y_x,data[i][4]));
	}
	sort(pt1D.begin(),pt1D.end());
	reverse(pt1D.begin(),pt1D.end());
	double min_etpy = DBL_MAX; 
	double threshold;
	for(int i=0; i<pt1D.size()-1; i++){
		double f = (pt1D[i].first+pt1D[i+1].first)/2;
		double pos0=0,neg0=0,pos1=0,neg1=0;
		for(int j=0; j<pt1D.size(); j++){
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

void Fisher_discriminant::precision_recall(vector<vector<double> > test_data){
	double threshold = this->threshold_calculation();
	for(int i=0; i<412; i++){
		double wx=0;
		for(int j=0; j<w.size(); j++){
			wx+=w[j]*test_data[i][j];
		}
		int predicted_class,true_class;
		if(wx<threshold){
			predicted_class=0;
		}
		else{
			predicted_class=1;
		}
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

void Fisher_discriminant::printOutput()
{
	cout << "TASK#1: FISHER DISCRIMINANT MODEL" << endl;
	cout << "  W_transpose is: [";
	for(int i=0; i<w.size(); i++) cout << w[i] << " ";
	cout << "]" << endl << endl;

	double accuracy = (double)(true_positive+true_negative)/(true_positive+
							true_negative+false_negative+false_positive);
	double precision = (double)true_positive/(true_positive+false_positive);
	double recall = (double)true_positive/(true_positive+false_negative);
	cout << "    Accuracy: " << accuracy << endl;
	cout << "    Precision: " << precision << endl;
	cout << "    Recall: " << recall << endl << endl;

	cout << "  CONFUSION MATRIX" << endl;
	cout << "\t      Predicted = 0\t  Predicted=1" << endl;
	cout << "    Actual=0: " << true_negative << "\t\t  " << false_positive << endl;
	cout << "    Actual=1: " << false_negative << "\t\t\t " << true_positive << endl << endl << endl;
}
