#define N 4

void getCofactor(double A[N][N], double temp[N][N], int p, int q, int n)
{
    int i = 0, j = 0;
    for (int row = 0; row < n; row++){
        for (int col = 0; col < n; col++){
            if (row != p && col != q){
                temp[i][j++] = A[row][col];
                if (j == n - 1){
                    j = 0;
                    i++;
                }
            }
        }
    }
}

double determinant(double A[N][N], int n)
{
    if (n == 1) return A[0][0];
    double D = 0, temp[N][N], sign = 1;
    for (int f = 0; f < n; f++){
        getCofactor(A, temp, 0, f, n);
        D += sign * A[0][f] * determinant(temp, n - 1);
        sign = -sign;
    }
    return D;
}
    
void adjoint(double A[N][N], double adj[N][N])
{
    if (N == 1){
        adj[0][0] = 1;
        return;
    }
    
    double sign = 1, temp[N][N];
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            getCofactor(A, temp, i, j, N);
            sign = ((i+j)%2==0)? 1: -1;
            adj[j][i] = (sign)*(determinant(temp, N-1));
        }
    }
}
    
bool inverse(double A[N][N], double inverse[N][N])
{
    double det = determinant(A, N);
    if (det == 0){
        cout << "Singular matrix, can't find its inverse";
        return false;
    }
    double adj[N][N];
    adjoint(A, adj);
    
    for (int i=0; i<N; i++)
        for (int j=0; j<N; j++)
            inverse[i][j] = adj[i][j]/det;
    
    return true;
}

void multiply(int n1, int m1, int n2, int m2, vector<vector<double> > &mat1, vector<vector<double> > &mat2, vector<vector<double> > &res)
{
    for(int i=0; i<n1; i++){
        for(int j=0; j<m2; j++){
            res[i][j] = 0.0;
            for(int k=0; k<m1; k++){
                res[i][j] += mat1[i][k]*mat2[k][j];
            }
        }
    }
}

double sigmoid(double value)
{
    return 1.0/(1.0+exp(-value));
}