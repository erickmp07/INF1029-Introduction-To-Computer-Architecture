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

        _mm256_store_ps(&matrix->rows[i], result);

        /*for(j=0;j<8;j++)
        {
        	*matrix->rows[j+i] = f[j]; 
        }*/
    }

    gettimeofday(&stop, NULL);

    elapsedTime = timedifference_msec(start, stop);

    printf("Scalar * matrix time: %f ms\n", elapsedTime);

    return OPERATION_OK;
}

int matrix_matrix_mult(
    struct matrix* a,
    struct matrix* b,
    struct matrix* c)
{
    unsigned long int NA, NB, NC, i, j, k, z, pos;
  float elapsedTime, *f;
    __m256 scnd_vector, frst_vector, result;

  struct timeval start, stop;

  gettimeofday(&start, NULL);

  /* Check the numbers of the elements of the matrix */
  NA = a->height * a->width;
  NB = b->height * b->width;
  NC = c->height * c->width;

  /* Check the integrity of the matrix */
  if ( (NA == 0 || a->rows == NULL) ||
       (NB == 0 || b->rows == NULL) ||
       (NC == 0 || c->rows == NULL) ) return OPERATION_ERROR;

  /* Check if we can execute de product of matrix A and matrib B */
  if ( (a->width != b->height) ||
       (c->height != a->height) ||
       (c->width != b->width) ) return OPERATION_ERROR;

  for (pos = 0; pos <  NC; ++pos) {
        i = pos / c->width;
        j = pos % c->width;

        c->rows[pos] = 0;

        /* Proccess the product between each element of the row of matrix a  */
        /* and each element of the colum of matrix b and accumulates the sum */
        /* of the product on the correspondent element of matrix c.          */
        for (k = 0; k < a->width; k+=8) {

            frst_vector = _mm256_load_ps(&a->rows[(i * a->width) + k]);

            scnd_vector = _mm256_loadu_ps(&b->rows[(k * b->height) + j]);

            result = _mm256_mul_ps(frst_vector, scnd_vector);

            f = (float*)&result;

            //_mm256_store_ps(float *a, result);

            for(z=0;z<8;z++)
            {
                c->rows[pos] += f[z];
            }

            //c->rows[pos] += a->rows[(i * a->width) + k] * b->rows[(k * b->height) + j];
        }
 }

  gettimeofday(&stop, NULL);

  elapsedTime = timedifference_msec(start, stop);

  printf("Matrix * matrix time: %f ms\n", elapsedTime);

    return OPERATION_OK;
}