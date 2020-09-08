#include "matrix_lib.h"

int main (void) 
{
	struct matrix *matrixA, *matrixB, *matrixC;

	matrixA = (struct matrix*)malloc(sizeof(struct matrix));
	matrixB = (struct matrix*)malloc(sizeof(struct matrix));
	matrixC = (struct matrix*)malloc(sizeof(struct matrix));

	matrixA->height = 3;
	matrixA->width = 2;
	matrixA->rows = (float *)malloc(matrixA->height*matrixA->width*sizeof(float));

	matrixB->height = 2;
	matrixB->width = 2;
	matrixB->rows = (float *)malloc(matrixB->height*matrixB->width*sizeof(float));

	matrixC->height = matrixA->height;
	matrixC->width = matrixB->width;
	matrixC->rows = (float *)malloc(matrixC->height*matrixC->width*sizeof(float));

	int i;

	float scalar = 2.0;

	printf("Matrix A\n");

	for(i = 1; i <= matrixA->height * matrixA->width; i++)
	{
		printf("%d\t", i);
		if(i % matrixA->width == 0)
		{
			printf("\n");
		}
		matrixA->rows[i - 1] = i;
	}

	printf("Matrix B\n");

	for(i = 1; i <= matrixB->height * matrixB->width; i++)
	{
		printf("%d\t", i);
		if(i % matrixB->width == 0)
		{
			printf("\n");
		}
		matrixB->rows[i - 1] = i;
	}

	printf("Matrix C\n");

	for(i = 1; i <= matrixC->height * matrixC->width; i++)
	{
		printf("%d\t", 0.0);
		if (i % matrixC->width == 0)
		{
			printf("\n");
		}

		matrixC->rows[i - 1] = 0.0;
	}

	i = matrix_matrix_mult(matrixA, matrixB, matrixC);

	if(i == OPERATION_OK)
	{
		printf("Operations was a success! Matrix C:\n");

		for(i = 0; i < matrixC->height * matrixC->width; i++)
		{
			if(i % matrixC->width == 0)
			{
				printf("\n");
			}

			printf("%.0f\t", matrixC->rows[i]);
		}
	}

	printf("\n");

	i = scalar_matrix_mult(scalar, matrixC);

	if(i == OPERATION_OK)
	{
		printf("Operations was a success! Matrix C:\n");

		for(i = 0; i < matrixC->height * matrixC->width; i++)
		{
			if(i % matrixC->width == 0)
			{
				printf("\n");
			}

			printf("%.0f\t", matrixC->rows[i]);
		}
	}

	return 0;
}