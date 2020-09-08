#include "matrix_lib.h"
#include <stdio.h>

const int OPERATION_OK = 1;
const int OPERATION_ERROR = 0;

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

    if (matrix->rows == NULL)
    {
        perror("Scalar * matrix couldn't be done because the rows are NULL.");
        return OPERATION_ERROR;
    }

    for (i = 0; i < matrix->height * matrix->width; i++)
    {
        matrix->rows[i] *= scalar_value;
    }

    return OPERATION_OK;
}

int matrix_matrix_mult(
    struct matrix* matrixA,
    struct matrix* matrixB,
    struct matrix* matrixC)
{
    int i;
    int j;
    int k;

    float product = 0.0;

    if (matrixA == NULL ||
        matrixB == NULL ||
        matrixC == NULL)
    {
        perror("Matrix * matrix couldn't be done because at least one of matrices is NULL.");
        return OPERATION_ERROR;
    }

    if (matrixA->rows == NULL ||
        matrixB->rows == NULL ||
        matrixC->rows == NULL)
    {
        perror("Matrix * matrix couldn't be done because the rows in at least one of matrices are NULL.");
        return OPERATION_ERROR;
    }

    if (matrixA->width != matrixB->height)
    {
        perror("Matrix * matrix couldn't be done because number of columns of A is different of number of rows of B.");
        return OPERATION_ERROR;
    }

    if (matrixA->height != matrixC->height ||
        matrixB->width != matrixC->width)
    {
        perror("Matrix * matrix couldn't be done because C dimensions are different of A * B result.");
        return OPERATION_ERROR;
    }

    for(i = 0; i < matrixA->height; i++)
    {
        for(k = 0; k < matrixB->width; k++)
        {
            product = 0.0;

            for( j = 0; j < matrixA->width; j++)
            {
                product = product + matrixA->rows[i * matrixA->width + j] * matrixB->rows[j * matrixB->width + k];
            }

            matrixC->rows[matrixB->width * i + k] = product;
        }
    }

    return OPERATION_OK;
}