#include "matrix_lib.h"
#include <string.h>

int main(int argc, char **argv)
{
	float scalar = atof(argv[1]);
	int matA_rows = atoi(argv[2]), matA_col = atoi(argv[3]), matB_rows = atoi(argv[4]), matB_col = atoi(argv[5]);
	char* matA_file = argv[6], *matB_file = argv[7], *res1_file = argv[8], *res2_file = argv[9];
	struct matrix *matrixA, *matrixB, *matrixC;

	matrixA = (struct matrix*)malloc(sizeof(struct matrix));
	matrixB = (struct matrix*)malloc(sizeof(struct matrix));
	matrixC = (struct matrix*)malloc(sizeof(struct matrix));

	matrixA->height = matA_col;
	matrixA->width = matA_rows;
	matrixA->rows = (float *)malloc(matrixA->height*matrixA->width*sizeof(float));

	matrixB->height = matB_col;
	matrixB->width = matB_rows;
	matrixB->rows = (float *)malloc(matrixB->height*matrixB->width*sizeof(float));

	matrixC->height = matrixA->height;
	matrixC->width = matrixB->width;
	matrixC->rows = (float *)malloc(matrixC->height*matrixC->width*sizeof(float));

	int i, count = 0;

	printf("File A: %s \n", matA_file);

	printf("Matrix A\n");

	FILE *file;
    file = fopen(matA_file,"r");
    if(!file)
    {
        perror("Error opening file");
        return -1;
    }

    while (!feof(file) &&(count < matA_rows * matA_col))
    {
        fscanf(file, "%f", &(matrixA->rows[count]));
        count++;
    }
    fclose(file);

	for(i = 1; i <= matrixA->height * matrixA->width; i++)
	{
		printf("%f\t", matrixA->rows[i - 1]);
		if(i % matrixA->width == 0)
		{
			printf("\n");
		}
	}

	printf("Matrix B\n");

	count = 0;

	file = fopen(matB_file,"r");
    if(!file)
    {
        perror("Error opening file");
        return -1;
    }

    while (!feof(file) &&(count < matB_rows * matB_col))
    {
        fscanf(file, "%f", &(matrixB->rows[count]));
        count++;
    }
    fclose(file);

	for(i = 1; i <= matrixB->height * matrixB->width; i++)
	{
		printf("%f\t", matrixB->rows[i - 1]);
		if(i % matrixB->width == 0)
		{
			printf("\n");
		}
	}

	printf("Matrix C\n");

	for(i = 1; i <= matrixC->height * matrixC->width; i++)
	{
		printf("%f\t", 0.0);
		if (i % matrixC->width == 0)
		{
			printf("\n");
		}

		matrixC->rows[i - 1] = 0.0;
	}

	printf("\n");

	i = scalar_matrix_mult(scalar, matrixA);

	file=fopen(res1_file,"wb");

	if (!file) 
		puts("Failed to open file");

	if(i == OPERATION_OK)
	{
		printf("Operations was a success! Matrix A:\n");

		for(i = 0; i < matrixA->height * matrixA->width; i++)
		{
			if(i % matrixA->width == 0)
			{
				printf("\n");
			}

			fprintf(file, "%.2f ",matrixA->rows[i]);
			printf("%.0f\t", matrixA->rows[i]);
		}
	}

	fclose(file);

	i = matrix_matrix_mult(matrixA, matrixB, matrixC);

	file=fopen(res2_file,"wb");

	if (!file) 
		puts("Failed to open file");

	if(i == OPERATION_OK)
	{
		printf("Operations was a success! Matrix C:\n");

		for(i = 0; i < matrixC->height * matrixC->width; i++)
		{
			if(i % matrixC->width == 0)
			{
				printf("\n");
			}

			fprintf(file, "%.2f ",matrixC->rows[i]);
			printf("%.0f\t", matrixC->rows[i]);
		}
	}

	fclose(file);

	return 0;
}