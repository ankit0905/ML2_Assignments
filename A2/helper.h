vector<pair<vector<double>, int> > loadDataset(string input_file)
{
    vector<pair<vector<double>, int> > data;
    ifstream infile(input_file);
    string token, delim = ",";
    string line;
    size_t pos = 0;
    int line_no = 0, index, target_value;
    double value;
    while(infile >> line){
        index = 0;
        vector<double> row;
        while((pos = line.find(delim)) != string::npos){
            token = line.substr(0, pos);
            value = atof(token.c_str());
            line.erase(0, pos + delim.length());
            row.push_back(value);
            index++;
        }
        target_value = atof(line.c_str());
        data.push_back(make_pair(row, target_value));
        line_no++;
    }
    return data;
}