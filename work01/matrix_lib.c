#include "matrix_lib.h"
#include <stdio.h>

int scalar_matrix_mult(
    float scalar_value,
    struct matrix *matrix)
{
    if (matrix == NULL)
    {
        perror("Scalar * matrix couldn't be done because the matrix is NULL->");
        return 0;
    }

    if (matrix->rows == NULL)
    {
        perror("Scalar * matrix couldn't be done because the rows are NULL");
        return 0;
    }

    int i;

    for (i = 0; i < matrix->width * matrix->height; i++)
    {
        matrix->rows[i] *= scalar_value;
    }

    return 1;
}

int matrix_matrix_mult(struct matrix *matrixA, struct matrix * matrixB, struct matrix * matrixC)
{
    if (matrixA == NULL || matrixB == NULL)
    {
        perror("Multiplication couldn't be done because one of the matrixes is NULL.");
        return 0;
    }

    if (matrixA->rows == NULL || matrixB->rows == NULL)
    {
        perror("Multiplication couldn't be done because the rows of one of the matrixes are NULL.");
        return 0;
    }

    matrixC->width = matrixB->width;

    matrixC->height = matrixA->height;

    matrixC->rows = (float *)malloc(matrixC->height*matrixC->width*sizeof(float));

    int i, j, z;
    float total;

    for(i=0;i<matrixA->height;i++)
    {
        for(z=0;z<matrixB->width; z++)
        {
            total = 0;
            for(j=0;j<matrixA->width; j++)
            {
                total = total + matrixA->rows[i * matrixA->width + j] * matrixB->rows[j*matrixB->width + z];
            }
            matrixC->rows[matrixB->width*i + z] = total;
        }
    }

    return 1;
}