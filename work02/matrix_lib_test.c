#include "matrix_lib.h"
#include <string.h>

int fillMatrixReadingFile(
	char* fileName,
	struct matrix* matrix)
{
	int count = 0;

	FILE *file;

    file = fopen(fileName, "r");
    if(file == NULL)
    {
        perror("Error opening file");
        return -1;
    }

    while (!feof(file) && (count < matrix->height * matrix->width))
    {
        fscanf(file, "%f", &(matrix->rows[count]));
        count++;
    }

    return fclose(file);
}

int fillMatrixWithConstant(
	float constant,
	struct matrix* matrix)
{
	int i;

	for(i = 0; i < matrix->height * matrix->width; i++)
	{
		matrix->rows[i] = constant;
	}

	return 0;
}

void printMatrix(
	char* matrixName,
	struct matrix* matrix)
{
	int i;

	printf(matrixName);

	for(i = 1; i <= matrix->height * matrix->width; i++)
	{
		printf("%.2f ", matrix->rows[i - 1]);
		/*if(i % matrix->width == 0)
		{
			printf("\n");
		}*/
		if(i == 256)
		{
			printf(" Ops. Max limit of print reached.\n\n");
			break;
		}
	}
}

int writeResult(
	char *fileName,
	struct matrix* matrix)
{
	int i;

	FILE *file = fopen(fileName, "wb");

	if (file == NULL)
	{ 
		perror("Failed to open file.");
		return -1;
	}

	for(i = 0; i < matrix->height * matrix->width; i++)
	{
		fprintf(file, "%.2f ",matrix->rows[i]);
	}

	return fclose(file);
}

void freeMatrix(
	struct matrix* matrix)
{
	free(matrix->rows);
	free(matrix);
}

int main(int argc, char **argv)
{
	float scalar = atof(argv[1]);
	int matA_rows = atoi(argv[2]), matA_col = atoi(argv[3]), matB_rows = atoi(argv[4]), matB_col = atoi(argv[5]);
	char* matA_file = argv[6], *matB_file = argv[7], *res1_file = argv[8], *res2_file = argv[9];
	struct matrix *matrixA, *matrixB, *matrixC;

	matrixA = (struct matrix*)malloc(sizeof(struct matrix));
	matrixB = (struct matrix*)malloc(sizeof(struct matrix));
	matrixC = (struct matrix*)malloc(sizeof(struct matrix));

	matrixA->height = matA_rows;
	matrixA->width = matA_col;
	//matrixA->rows = (float *)malloc(matrixA->height * matrixA->width * sizeof(float));
	matrixA->rows =  (float*)aligned_alloc(32, matrixA->height * matrixA->width * sizeof(float));

	matrixB->height = matB_rows;
	matrixB->width = matB_col;
	//matrixB->rows = (float *)malloc(matrixB->height * matrixB->width * sizeof(float));
	matrixB->rows =  (float*)aligned_alloc(32, matrixB->height * matrixB->width * sizeof(float));

	matrixC->height = matrixA->height;
	matrixC->width = matrixB->width;
	//matrixC->rows = (float *)malloc(matrixC->height * matrixC->width * sizeof(float));
	matrixC->rows =  (float*)aligned_alloc(32, matrixC->height * matrixC->width * sizeof(float));

	int i;

	printf("File A: %s \n", matA_file);

	fillMatrixReadingFile(matA_file, matrixA);
	//printMatrix("Matrix A\n", matrixA);

	fillMatrixReadingFile(matB_file, matrixB);
	//printMatrix("Matrix B\n", matrixB);

	fillMatrixWithConstant(0.0, matrixC);
	//printMatrix("Matrix C\n", matrixC);

	printf("\n");

	i = scalar_matrix_mult(scalar, matrixA);
	//printMatrix("Matrix A\n", matrixA);

	if(i == OPERATION_OK)
	{
		writeResult(res1_file, matrixA);
	}

	i = matrix_matrix_mult(matrixA, matrixB, matrixC);
	//printMatrix("Matrix C\n", matrixC);

	if(i == OPERATION_OK)
	{
		//printf("Operations was a success! Matrix C:\n");

		writeResult(res2_file, matrixC);
	}

	freeMatrix(matrixA);
	freeMatrix(matrixB);
	freeMatrix(matrixC);

	return 0;
}