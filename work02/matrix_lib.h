#ifndef MATRIX_LIB_H
#define MATRIX_LIB_H

#include <immintrin.h>
#include<stdio.h>
#include<stdlib.h>
#include "timer.h"

const int OPERATION_OK;
const int OPERATION_ERROR;

struct matrix {
    unsigned long int height;
    unsigned long int width;
    float *rows;
};

int scalar_matrix_mult(
    float scalar_value,
    struct matrix *matrix);

int matrix_matrix_mult(
    struct matrix *matrixA,
    struct matrix *matrixB,
    struct matrix *matrixC);

#endif