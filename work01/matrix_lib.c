#include "matrix_lib"

int scalar_matrix_mult(
    float scalar_value,
    struct matrix *matrix)
{
    int i;

    if (matrix == NULL)
    {
        perror("Scalar * matrix couldn't be done because the matrix is NULL.");
        return OPERATION_ERROR;
    }

    if (matrix.rows == NULL)
    {
        perror("Scalar * matrix couldn't be done because the rows are NULL.");
        return OPERATION_ERROR;
    }

    for (i = 0; i < matrix.height * matrix.width; i++)
    {
        matrix.rows[i] *= scalar_value;
    }

    return OPERATION_OK;
}

/*int matrix_matrix_mult(
    struct matrix* A,
    struct matrix* B,
    struct matrix* C)
{
    int i;
    int j;
    int k;

    float product = 0.0;

    if (A == NULL ||
        B == NULL ||
        C == NULL)
    {
        perror("Matrix * matrix couldn't be done because at least one of matrices is NULL.");
        return OPERATION_ERROR;
    }

    if (A.rows == NULL ||
        B.rows == NULL ||
        C.rows == NULL)
    {
        perror("Matrix * matrix couldn't be done because the rows in at least one of matrices are NULL.");
        return OPERATION_ERROR;
    }

    if (A.width != B.height)
    {
        perror("Matrix * matrix couldn't be done because number of columns of A is different of number of rows of B.");
        return OPERATION_ERROR;
    }

    if (A.height != C.height ||
        B.width != C.width)
    {
        perror("Matrix * matrix couldn't be done because C dimensions are different of A * B result.")
    }

    for (i = 0; i < A.height; i++)
    {
        for (j = 0; j < B.width; j++)
        {
            for (k = 0; k < A.width; k++)
            {
                product += A.rows[i * A.width + k] + B.rows[j * B.height + k];
            }

            C.rows[i * A.width + j] = product;
            product = 0.0;
        }
    }
}*/