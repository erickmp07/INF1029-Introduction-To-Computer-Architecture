#include "matrix_lib.h"

int main (void) 
{
	struct matrix *matrixA, *matrixB, *matrixC;

	matrixA = (struct matrix*)malloc(sizeof(struct matrix));
	matrixB = (struct matrix*)malloc(sizeof(struct matrix));
	matrixC = (struct matrix*)malloc(sizeof(struct matrix));

	matrixA->height = 2;
	matrixA->width = 2;
	matrixA->rows = (float *)malloc(matrixA->height*matrixA->width*sizeof(float));

	matrixB->height = 2;
	matrixB->width = 4;
	matrixB->rows = (float *)malloc(matrixB->height*matrixB->width*sizeof(float));

	int i;

	float scalar = 2.0;

	printf("Matrix A\n");

	for(i = 1; i <= matrixA->height * matrixA->width; i++)
	{
		printf("%d ", i);
		if(i % matrixA->width == 0)
		{
			printf("\n");
		}
		matrixA->rows[i-1] = i;
	}

	printf("Matrix B\n");

	for(i = 1; i <= matrixB->height*matrixB->width; i++)
	{
		printf("%d ", i);
		if(i % matrixB->width == 0)
		{
			printf("\n");
		}
		matrixB->rows[i-1] = i;
	}

	i = matrix_matrix_mult(matrixA, matrixB, matrixC);

	if(i == OPERATION_OK)
	{
		printf("Operations was a success! Matrix C:\n");

		for(i = 0; i <= matrixC->height * matrixC->width; i++)
		{
			if(i % matrixC->width == 0)
			{
				printf("\n");
			}

			printf("%.0f ", matrixC->rows[i]);
		}
	}

	i = scalar_matrix_mult(scalar, matrixC);

	if(i == OPERATION_OK)
	{
		printf("Operations was a success! Matrix C:\n");

		for(i = 0; i <= matrixC->height * matrixC->width; i++)
		{
			if(i % matrixC->width == 0)
			{
				printf("\n");
			}

			printf("%.0f ", matrixC->rows[i]);
		}
	}

	return 0;
}