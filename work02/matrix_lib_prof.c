#include "matrix_lib.h"
#include <immintrin.h>

int scalar_matrix_mult(float scalar_value, struct matrix *matrix) {
  unsigned long int i;
  unsigned long int N;
  float elapsedTime, *f;

  struct timeval start, stop;

  gettimeofday(&start, NULL);

  /* Check the numbers of the elements of the matrix */
  N = matrix->height * matrix->width;

  /* Check the integrity of the matrix */
  if (N == 0 || matrix->rows == NULL) return 0;

  for (i = 0; i < N; ++i) {
        matrix->rows[i] = matrix->rows[i] * scalar_value;
  }

  gettimeofday(&stop, NULL);

  elapsedTime = timedifference_msec(start, stop);

  printf("Scalar * matrix time: %f ms\n", elapsedTime);

  return 1;
}

int matrix_matrix_mult(struct matrix *a, struct matrix *b, struct matrix *c) {
  unsigned long int NA, NB, NC, i, j, k, pos;
  float elapsedTime, *f;

  struct timeval start, stop;

  gettimeofday(&start, NULL);

  /* Check the numbers of the elements of the matrix */
  NA = a->height * a->width;
  NB = b->height * b->width;
  NC = c->height * c->width;

  /* Check the integrity of the matrix */
  if ( (NA == 0 || a->rows == NULL) ||
       (NB == 0 || b->rows == NULL) ||
       (NC == 0 || c->rows == NULL) ) return 0;

  /* Check if we can execute de product of matrix A and matrib B */
  if ( (a->width != b->height) ||
       (c->height != a->height) ||
       (c->width != b->width) ) return 0;

  for (pos = 0; pos <  NC; ++pos) {
        i = pos / c->width;
        j = pos % c->width;

        c->rows[pos] = 0;

        /* Proccess the product between each element of the row of matrix a  */
        /* and each element of the colum of matrix b and accumulates the sum */
        /* of the product on the correspondent element of matrix c.          */
        for (k = 0; k < a->width; ++k) {
                c->rows[pos] += a->rows[(i * a->width) + k] * b->rows[(k * b->height) + j];
        }
 }

  gettimeofday(&stop, NULL);

  elapsedTime = timedifference_msec(start, stop);

  printf("Matrix * matrix time: %f ms\n", elapsedTime);

  return 1;
}
