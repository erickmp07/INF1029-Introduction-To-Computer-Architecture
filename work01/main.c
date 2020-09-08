#include "matrix_lib.h"
#include<conio.h>
#include<stdlib.h>

void libera(
    struct matrix* A)
{
    free(A->rows);

    free(A);
}

int main(void)
{
    int heightA = 3;
    int widthA = 2;
    int heightB = 2;
    int widthB = 2;

    struct matrix* A;
    struct matrix* B;
    struct matrix* C;

    A->height = heightA;
    A->width = widthA;
    A->rows = (float *)malloc(A->height * A->width * sizeof(float));

    A->rows[0] = 2;
    A->rows[1] = 4;
    A->rows[2] = 7;
    A->rows[3] = 6;
    A->rows[4] = 8;
    A->rows[5] = 9;

    for (int i = 0; i < A->height * A->width; i++)
    {
        printf("%f", A->rows[i]);
        printf("\t");
    }

    B->height = heightB;
    B->width = widthB;
    B->rows = (float *)malloc(B->height * B->width * sizeof(float));

    B->rows[0] = 1;
    B->rows[1] = 2;
    B->rows[2] = 2;
    B->rows[3] = 1;

    C->height = heightA;
    C->width = widthB;
    C->rows = (float *)malloc(C->height * C->width * sizeof(float));

    libera(A);
    libera(B);
    libera(C);

    getch();

    return 0;
}