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

int matrix_matrix_mult(struct matrix *a, struct matrix *b, struct matrix *c) {

	unsigned long int NA, NB, NC, c_line, a_col, b_col, z;
	float elapsedTime, *f, *first_c_i_j, *next_a_i_j, *next_b_i_j, *next_c_i_j, *next_c_i_j_aux;
	__m256 a_vector, b_vector, c_vector, result;

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

			for (c_line = 0; c_line < c->height; ++c_line) {
				first_c_i_j = c->rows + (c_line * c->width);
				next_a_i_j = a->rows + (c_line * a->width);
				next_b_i_j = b->rows;
				for (a_col = 0; a_col < a->width; ++a_col, ++next_a_i_j) { 
					next_c_i_j = first_c_i_j;
					for (b_col = 0; b_col < b->width; b_col+=8) {
						if (a_col == 0)
						{
							next_c_i_j_aux = next_c_i_j;
							for(z=0;z<8;z++, next_c_i_j_aux++)
							{
								*(next_c_i_j_aux) = 0.0f;
							}
						} 

						a_vector = _mm256_setr_ps(*next_a_i_j, *next_a_i_j, *next_a_i_j, *next_a_i_j, *next_a_i_j, *next_a_i_j, *next_a_i_j, *next_a_i_j);

						b_vector = _mm256_load_ps(next_b_i_j);

						result = _mm256_mul_ps(a_vector, b_vector);

						f = (float*)&result;

						for(z=0;z<8;z++, ++next_c_i_j)
						{
							*(next_c_i_j) += f[z];
							next_b_i_j++;
						}
					}

				}
			}

			gettimeofday(&stop, NULL);

			elapsedTime = timedifference_msec(start, stop);

			printf("Matrix * matrix time: %f ms\n", elapsedTime);

			return OPERATION_OK;
		}