#include "matrix_lib.h"

int global_n_block = 4096;
int global_n_thread = 256;
cudaError_t cudaError;

__global__ 
void scalar_aux(int fim, float *rows, float scalar_value)
{
  int i;
  int ini = blockIdx.x * blockDim.x + threadIdx.x;
  int passo = blockDim.x * gridDim.x;

  for (i = ini; i < fim; i += passo) 
  {
    rows[i] = rows[i] * scalar_value;
  }
}

int scalar_matrix_mult(float scalar_value, struct matrix *matrix)
{
  if(matrix == NULL)
  {
    return ERRO;
  }

  int size = matrix->height * matrix->width;
  int i;

  int end = (size + DATASET_SIZE - 1) / DATASET_SIZE;
  int tam = DATASET_SIZE;

  for(i = 0; i < end; ++i)
  {
    if(size % DATASET_SIZE != 0 && i == end - 1)
    {
      tam = size % DATASET_SIZE;
    }
    
    cudaError = cudaMemcpy(matrix->d_rows, matrix->h_rows+(i*tam), tam*sizeof(float), cudaMemcpyHostToDevice);

    if (cudaError != cudaSuccess) 
    {
      printf("cudaMemcpy (h -> d) returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
      return ERRO;
    }

    int blockSize = global_n_thread;
    int numBlocks = (tam + blockSize - 1) / blockSize;
    
    if (numBlocks > global_n_block)
    {
      numBlocks = global_n_block;
    }

    scalar_aux<<<numBlocks, blockSize>>>(tam, matrix->d_rows, scalar_value);

    cudaDeviceSynchronize();

    cudaError = cudaMemcpy(matrix->h_rows+(i*tam), matrix->d_rows, tam*sizeof(float), cudaMemcpyDeviceToHost);
  
    if (cudaError != cudaSuccess)
    {
      printf("cudaMemcpy (d -> h) returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
      
      return ERRO;
    }
  }

  return SUCESSO;
}

int matrix_matrix_mult(struct matrix *matrixA, struct matrix *matrixB, struct matrix *matrixC)
{
    return SUCESSO;
}

int set_grid_size(int threads_per_block, int max_blocks_per_grid)
{
  if(threads_per_block > MAX_THREAD ||
     max_blocks_per_grid > MAX_BLOCK) 
  {
    return ERRO;
  }

  global_n_block = threads_per_block;
  global_n_thread = max_blocks_per_grid;
  
  return SUCESSO;
}