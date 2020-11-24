struct matrix
{
    unsigned long int height;
    unsigned long int width;
    float *h_rows;
    float *d_rows;
};

int scalar_matrix_mult(float scalar_value, struct matrix *matrix)
{
    return 1;
}