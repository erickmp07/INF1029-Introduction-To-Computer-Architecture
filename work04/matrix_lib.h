#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

#define SUCESSO 1
#define ERRO 0
#define MAX_THREAD 1024
#define MAX_BLOCK 65535
#define DATASET_SIZE 1048576

struct matrix
{
    unsigned long int height;
    unsigned long int width;
    float *h_rows;
    float *d_rows;
};

int scalar_matrix_mult(float scalar_value, struct matrix *matrix);

int matrix_matrix_mult(struct matrix *matrixA, struct matrix *matrixB, struct matrix *matrixC);

int set_grid_size(int threads_per_block, int max_blocks_per_grid);