class LogisticRegression
{
    private:
        vector<vector<double> > data;
        vector<double> weights;
        int rows, dimensions;
        int true_negative = 0, true_positive = 0, false_negative = 0, false_positive = 0;
        double alpha;

    public:
        LogisticRegression(vector<vector<double> > training_data);
        double calculateCost(vector<double> params);
        double calculateChange(vector<double> params, int idx);
        int gradientDescent(double alpha, double epsilon);
        void predict(vector<vector<double> > test_data);
        void printOutput();
};

LogisticRegression::LogisticRegression(vector<vector<double> > training_data)
{
    data = training_data;
    rows = data.size();
    dimensions = data[0].size();
    for(int i=0; i<dimensions; i++) weights.push_back(1.0);
    gradientDescent(0.02, 0.001);
}

double LogisticRegression::calculateCost(vector<double> params)
{
    double cost = 0.0, sum;
    for(int i=0; i<rows; i++){
        sum = 0.0;
        for(int j=0; j<dimensions; j++){
            if(j == 0) sum += params[0];
            else sum += params[j]*data[i][j-1];
        }
        sum = sigmoid(sum);
        if(data[i][dimensions-1] == 0)
            cost += -1.0*(1-sum);
        else 
            cost += -1.0*sum;
    }
    return cost;
}

double LogisticRegression::calculateChange(vector<double> params, int idx)
{
    double tot = 0.0, yi, ti;
    for(int i=0; i<rows; i++){
        ti = data[i][dimensions-1];
        yi = 0.0;
        for(int j=0; j<dimensions; j++){
            if(j == 0) yi += params[0];
            else yi += params[j]*data[i][j-1];
        }
        yi = sigmoid(yi);
        if(idx == 0) tot += (yi-ti);
        else tot += (yi-ti)*data[i][idx-1];
    }
    return tot;
}

int LogisticRegression::gradientDescent(double alpha, double epsilon)
{
    int iterations = 0;
    double prev_cost, curr_cost = calculateCost(weights), delta_cost = 1.0, prev;
    vector<double> temp = weights;
    while(delta_cost > epsilon){
        prev_cost = curr_cost;
        for(int i=0; i<dimensions; i++){
            temp[i] = temp[i] - alpha * calculateChange(weights, i);
        }
        weights = temp;
        curr_cost = calculateCost(weights);
        delta_cost = prev_cost - curr_cost;
        iterations++;
    }
    return iterations;
}

void LogisticRegression::predict(vector<vector<double> > test_data)
{
    int predicted_class, true_class;
    double value;
    for(int i=0; i<412; i++){
        value = 0.0;
        for(int j=0; j<dimensions; j++){
            if(j == 0) value += weights[0];
            else value += weights[j]*test_data[i][j-1];
        }
        value = sigmoid(value);
        
        true_class = test_data[i][dimensions-1];
        if(value <= 0.5) predicted_class = 0;
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

void LogisticRegression::printOutput()
{
    cout << "TASK#3: LOGISTIC REGRESSION (DISCRIMINATIVE MODEL)" << endl;
    cout << "  W_transpose is: [";
	for(int i=0; i<weights.size(); i++) cout << weights[i] << " ";
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
	cout << "    Actual=0: " << true_negative << "\t\t\t  " << false_positive << endl;
	cout << "    Actual=1: " << false_negative << "\t\t\t " << true_positive << endl;
}