#include "matrix_lib.h"

const int OPERATION_OK = 1;
const int OPERATION_ERROR = 0;

int scalar_matrix_mult(
    float scalar_value,
    struct matrix *matrix)
{
    int i;
    float elapsedTime;

    struct timeval start, stop;

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

    gettimeofday(&start, NULL);

    for (i = 0; i < matrix->height * matrix->width; i++)
    {
        matrix->rows[i] *= scalar_value;
    }

    gettimeofday(&stop, NULL);

    elapsedTime = timedifference_msec(start, stop);

    printf("Scalar * matrix time: %f ms\n", elapsedTime);

    return OPERATION_OK;
}

int matrix_matrix_mult(
    struct matrix* matrixA,
    struct matrix* matrixB,
    struct matrix* matrixC)
{
    int i = 0;
    int j = 0;
    int k = 0;
    int indexLineMatrixA = 0;
    int indexColumnMatrixB = 0;

    float elapsedTime;

    struct timeval start, stop;

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

    gettimeofday(&start, NULL);

    for(i = 0; i < matrixA->height * matrixA->width; i++)
    {
        indexLineMatrixA = i / matrixA->width;

        k = i % matrixA->width == 0
            ? 0
            : k + 1;

        for(j = k * matrixB->width; j < matrixB->width * (k + 1); j++)
        {
            indexColumnMatrixB = j % matrixB->width;

            matrixC->rows[(indexLineMatrixA * matrixA->width) + indexColumnMatrixB] += matrixA->rows[i] * matrixB->rows[j];
        }
    }

    gettimeofday(&stop, NULL);

    elapsedTime = timedifference_msec(start, stop);

    printf("Matrix * matrix time: %f ms\n", elapsedTime);

    return OPERATION_OK;
}