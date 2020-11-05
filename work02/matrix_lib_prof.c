#include <immintrin.h>

struct matrix {
	unsigned long int height;
	unsigned long int width;
	float *rows;
};

int scalar_matrix_mult(float scalar_value, struct matrix *matrix) {
  unsigned long int i;
  unsigned long int N;

  /* Check the numbers of the elements of the matrix */
  N = matrix->height * matrix->width;

  /* Check the integrity of the matrix */
  if (N == 0 || matrix->rows == NULL) return 0;

  for (i = 0; i < N; ++i) {
        matrix->rows[i] = matrix->rows[i] * scalar_value;
  }

  return 1;
}

int matrix_matrix_mult(struct matrix *a, struct matrix *b, struct matrix *c) {
  unsigned long int NA, NB, NC, c_line, a_col, b_col;
  float *first_c_i_j, *next_a_i_j, *next_b_i_j, *next_c_i_j;

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

  for (c_line = 0; c_line < c->height; ++c_line) {
	first_c_i_j = c->rows + (c_line * c->width);
	next_a_i_j = a->rows + (c_line * a->width);
	next_b_i_j = b->rows;
	for (a_col = 0; a_col < a->width; ++a_col, ++next_a_i_j) { 
		next_c_i_j = first_c_i_j;
		for (b_col = 0; b_col < b->width; ++b_col, ++next_b_i_j, ++next_c_i_j) {
			if (a_col == 0) *(next_c_i_j) = 0.0f;
			*(next_c_i_j) += *(next_a_i_j) * *(next_b_i_j);
		}
			
	}
  }

  return 1;
}
