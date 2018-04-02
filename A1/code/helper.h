#define N 4

/* Function to get the cofactor matrix for mat[p][q]
    in res[][].
        mat: input matrix
        res: output matrix
        p, q: element for which cofactor is calculated.
*/
void getCofactor(double mat[N][N], double res[N][N], int p, int q, int n)
{
    int i = 0, j = 0;
    for (int r = 0; r < n; r++){
        for (int c = 0; c < n; c++){
            if (r != p && c != q){
                res[i][j++] = mat[r][c];
                if (j == n - 1){
                    j = 0;
                    i++;
                }
            }
        }
    }
}

/* Recursive function for finding determinant of a matrix
    mat: input matrix
    n: dimension of the square matrix
*/
double determinant(double mat[N][N], int n)
{
    if (n == 1) return mat[0][0];
    double det = 0, res[N][N], sign = 1;
    for (int i = 0; i < n; i++){
        getCofactor(mat, res, 0, i, n);
        det += sign * mat[0][i] * determinant(res, n - 1);
        sign = -1.0 * sign;
    }
    return det;
}

/* Function to compute the adjoint of given matrix.
        mat: input matrix
        adj: resultant adjoint matrix to be computed
*/
void adjoint(double mat[N][N], double adj[N][N])
{
    if (N == 1){
        adj[0][0] = 1;
        return;
    }

    double sign = 1, res[N][N];
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            getCofactor(mat, res, i, j, N);
            if((i+j)%2 == 0) sign = 1;
            else sign = -1;
            adj[j][i] = sign*determinant(res, N-1);
        }
    }
}
    
/* Function to compute the inverse of the given matrix.
        mat: input matrix
        inverse: resultant inverse matrix to be computed
*/
bool inverse(double mat[N][N], double inverse[N][N])
{
    double det = determinant(mat, N);
    if (det == 0){
        cout << "Inverse does not exist";
        return false;
    }
    double adj[N][N];
    adjoint(mat, adj);
    
    for (int i=0; i<N; i++)
        for (int j=0; j<N; j++)
            inverse[i][j] = adj[i][j]/det;
    
    return true;
}

/* Function to multiply two matrices.
        n1,m1: dimensions of first matrix
        n2,m2: dimensions of second matrix
        mat1, mat2: the two input matrices
        res: the resultant matrix to be computed
*/
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

/* Function to compute sigmoid of a value.
        value: input to sigmoid function
*/
double sigmoid(double value)
{
    return 1.0/(1.0+exp(-value));
}