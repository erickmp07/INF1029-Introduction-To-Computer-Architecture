#include "matrix_lib"

int scalar_matrix_mult(
    float scalar_value,
    struct matrix *matrix)
{
    if (matrix == NULL)
    {
        perror("Scalar * matrix couldn't be done because the matrix is NULL.");
        return 0;
    }

    if (matrix.rows == NULL)
    {
        perror("Scalar * matrix couldn't be done because the rows are NULL");
        return 0;
    }

    int i;

    for (i = 0; i < matrix.width * matrix.height; i++)
    {
        matrix.rows[i] *= scalar_value;
    }

    return 1;
}