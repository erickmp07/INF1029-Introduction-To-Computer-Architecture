#include "matrix_lib.h"

#define THREADS_PER_BLOCK_DEFAULT 256
#define THREADS_PER_BLOCK 1024
#define MAX_BLOCKS_PER_GRID_DEFAULT 4096
#define MAX_BLOCKS_PER_GRID 65535

int blockSize = THREADS_PER_BLOCK_DEFAULT;
int numBlocks = MAX_BLOCKS_PER_GRID_DEFAULT;

int scalar_matrix_mult(float scalar_value, struct matrix *matrix)
{
    return SUCESSO;
}

int matrix_matrix_mult(struct matrix *matrixA, struct matrix *matrixB, struct matrix *matrixC)
{
    return SUCESSO;
}

int set_grid_size(int threads_per_block, int max_blocks_per_grid)
{
    if (threads_per_block > THREADS_PER_BLOCK || 
        max_blocks_per_grid > MAX_BLOCKS_PER_GRID)
    {
        return ERRO;
    }

    blockSize = threads_per_block;
    numBlocks = max_blocks_per_grid;

    return SUCESSO;
}