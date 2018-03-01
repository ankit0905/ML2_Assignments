#define N 4

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
        getCofactor(A, temp, 0, f, n);
        D += sign * A[0][f] * determinant(temp, n - 1);
    
        // terms are to be added with alternate sign
        sign = -sign;
    }
    
    return D;
}
    
// Function to get adjoint of A[N][N] in adj[N][N].
void adjoint(double A[N][N], double adj[N][N])
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
            getCofactor(A, temp, i, j, N);
    
            // sign of adj[j][i] positive if sum of row
            // and column indexes is even.
            sign = ((i+j)%2==0)? 1: -1;
    
            // Interchanging rows and columns to get the
            // transpose of the cofactor matrix
            adj[j][i] = (sign)*(determinant(temp, N-1));
        }
    }
}
    
// Function to calculate and store inverse, returns false if
// matrix is singular
bool inverse(double A[N][N], double inverse[N][N])
{
    // Find determinant of A[][]
    double det = determinant(A, N);
    if (det == 0){
        cout << "Singular matrix, can't find its inverse";
        return false;
    }
    
    // Find adjoint
    double adj[N][N];
    adjoint(A, adj);
    
    // Find Inverse using formula "inverse(A) = adj(A)/det(A)"
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