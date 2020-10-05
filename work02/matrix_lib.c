#include "matrix_lib.h"
#include <immintrin.h>

const int OPERATION_OK = 1;
const int OPERATION_ERROR = 0;

int scalar_matrix_mult(
    float scalar_value,
    struct matrix *matrix)
{
    int i, j;
    float elapsedTime, *f;
    __m256 vector, scalar_vector, result;

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

    for (i = 0; i < matrix->height * matrix->width; i=i+8)
    {
    	vector = _mm256_load_ps(&matrix->rows[i]);

    	scalar_vector = _mm256_setr_ps(scalar_value, scalar_value, scalar_value, scalar_value, scalar_value, scalar_value, scalar_value, scalar_value);

    	result = _mm256_mul_ps(vector, scalar_vector);

        //matrix->rows[i] *= scalar_value;

        f = (float*)&result;

        for(j=0;j<8;j++)
        {
        	matrix->rows[j+i] = f[j]; 
        }
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
    int z;
    int indexLineMatrixA = 0;
    int indexColumnMatrixB = 0;
    float* f;
    __m256 frst_vector, scnd_vector, result;

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

        frst_vector = _mm256_setr_ps(matrixA->rows[i], matrixA->rows[i], matrixA->rows[i], matrixA->rows[i], matrixA->rows[i], matrixA->rows[i], matrixA->rows[i], matrixA->rows[i]);
        for(j = k * matrixB->width; j < matrixB->width * (k + 1); j +=8)
        {
    		scnd_vector = _mm256_load_ps(&matrixB->rows[j]);

    		result = _mm256_mul_ps(frst_vector, scnd_vector);

            f = (float*)&result;

    		for(z = 0; z < 8; z++)
    		{
            	indexColumnMatrixB = (j - 8 + z) % matrixB->width;
            	matrixC->rows[(indexLineMatrixA * matrixA->width) + indexColumnMatrixB] += f[z];
    		}
        }
    }

    gettimeofday(&stop, NULL);

    elapsedTime = timedifference_msec(start, stop);

    printf("Matrix * matrix time: %f ms\n", elapsedTime);

    return OPERATION_OK;
}