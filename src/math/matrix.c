/*
 * Neural Network From Scratch in C
 *
 * Copyright (C) 2026 Sanskar Rai 
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "common.h"
#include "math/matrix.h"
#include "error/error.h"

#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <assert.h>

/*==============================================================================
 * Creation & Destruction
 *============================================================================*/

Error matrix_create(Matrix *mat, usize rows, usize cols) {
    assert(mat);
    assert(rows > 0);
    assert(cols > 0);

    float *data = malloc(rows * cols * sizeof(float));
    if (!data)
        return ERROR_ALLOCATION_FAILED;

    mat->rows = rows;
    mat->cols = cols;
    mat->data = data;

    return ERROR_OK;
}

Error matrix_copy_buffer(Matrix *mat, usize rows, usize cols, float *buffer) {
    Error err;

    assert(mat);
    assert(buffer);
    assert(rows > 0);
    assert(cols > 0);

    err = matrix_create(mat, rows, cols);
    if (err != ERROR_OK)
        return err;

    memcpy(mat->data, buffer, rows * cols * sizeof(float));
    return ERROR_OK;
}

void matrix_destroy(Matrix *mat) {
    assert(mat);

    free(mat->data);
    mat->data = NULL;
    mat->rows = 0;
    mat->cols = 0;
}

/*==============================================================================
 * Element Access
 *============================================================================*/

float matrix_get(const Matrix *mat, usize row, usize col) {
    assert(mat);
    assert(row < mat->rows);
    assert(col < mat->cols);

    return mat->data[row * mat->cols + col];
}

void matrix_set(Matrix *mat, usize row, usize col, float value) {
    assert(mat);
    assert(row < mat->rows);
    assert(col < mat->cols);

    mat->data[row * mat->cols + col] = value;
}

/*==============================================================================
 * Utility
 *============================================================================*/

usize matrix_size(const Matrix *mat) {
    assert(mat);
    return mat->rows * mat->cols;
}

Matrix matrix_empty(void) {
    return (Matrix) {
        .rows = 0,
        .cols = 0,
        .data = NULL
    };
}

void matrix_fill(Matrix *mat, float value) {
    assert(mat);
    assert(mat->data);

    usize n = matrix_size(mat);
    for (usize i = 0; i < n; i++) {
        mat->data[i] = value;
    }
}

void matrix_copy(Matrix *dest, const Matrix *src) {
    assert(dest);
    assert(src);
    assert(src->data);
    assert(dest->rows == src->rows);
    assert(dest->cols == src->cols);
    assert(dest->data);

    memcpy(dest->data, src->data, matrix_size(src) * sizeof(float));
}

/*==============================================================================
 * Random Initialization
 *============================================================================*/

static inline float random_float(float min, float max) {
    static bool seeded = false;
    if (!seeded) {
        srand(time(NULL));
        seeded = true;
    }

    float scale = (float)rand() / (float)RAND_MAX;
    return min + scale * (max - min);
}

void matrix_randomize(Matrix *mat, float min, float max) {
    assert(mat);
    assert(mat->data);

    usize n = matrix_size(mat);
    for (usize i = 0; i < n; i++) {
        mat->data[i] = random_float(min, max);
    }
}

void matrix_he_uniform(Matrix *mat, usize fan_in) {
    assert(mat);
    assert(mat->data);
    assert(fan_in > 0);

    float limit = sqrtf(6.0f / (float)fan_in);
    matrix_randomize(mat, -limit, limit);
}

void matrix_xavier_uniform(Matrix *mat, usize fan_in, usize fan_out) {
    assert(mat);
    assert(mat->data);
    assert(fan_in + fan_out > 0);

    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    matrix_randomize(mat, -limit, limit);
}

/*==============================================================================
 * Matrix Arithmetic
 *============================================================================*/

void matrix_add(Matrix *result, const Matrix *a, const Matrix *b) {
    assert(result);
    assert(a);
    assert(b);
    assert(a->rows == b->rows);
    assert(a->cols == b->cols);
    assert(result->rows == a->rows);
    assert(result->cols == a->cols);

    usize n = matrix_size(a);
    for (usize i = 0; i < n; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
}

void matrix_subtract(Matrix *result, const Matrix *a, const Matrix *b) {
    assert(result);
    assert(a);
    assert(b);
    assert(a->rows == b->rows);
    assert(a->cols == b->cols);
    assert(result->rows == a->rows);
    assert(result->cols == a->cols);

    usize n = matrix_size(a);
    for (usize i = 0; i < n; i++) {
        result->data[i] = a->data[i] - b->data[i];
    }
}

void matrix_multiply(Matrix *result, const Matrix *a, const Matrix *b) {
    assert(result);
    assert(a);
    assert(b);
    assert(a->cols == b->rows);
    assert(result->rows == a->rows);
    assert(result->cols == b->cols);
    assert(result->data != a->data);
    assert(result->data != b->data);

    memset(result->data, 0, matrix_size(result) * sizeof(float));

    for (usize i = 0; i < a->rows; i++) {
        for (usize k = 0; k < a->cols; k++) {
            float a_val = matrix_get(a, i, k);
            for (usize j = 0; j < b->cols; j++) {
                result->data[i * result->cols + j] += a_val * matrix_get(b, k, j);
            }
        }
    }
}

void matrix_hadamard(Matrix *result, const Matrix *a, const Matrix *b) {
    assert(result);
    assert(a);
    assert(b);
    assert(a->rows == b->rows);
    assert(a->cols == b->cols);
    assert(result->rows == a->rows);
    assert(result->cols == a->cols);

    usize n = matrix_size(a);
    for (usize i = 0; i < n; i++) {
        result->data[i] = a->data[i] * b->data[i];
    }
}

void matrix_scalar_multiply(Matrix *result, const Matrix *mat, float scalar) {
    assert(result);
    assert(mat);
    assert(mat->data);
    assert(result->rows == mat->rows);
    assert(result->cols == mat->cols);

    usize n = matrix_size(mat);
    for (usize i = 0; i < n; i++) {
        result->data[i] = mat->data[i] * scalar;
    }
}

void matrix_transpose(Matrix *result, const Matrix *mat) {
    assert(result);
    assert(mat);
    assert(mat->data);
    assert(result->rows == mat->cols);
    assert(result->cols == mat->rows);
    assert(result->data != mat->data);

    for (usize i = 0; i < mat->rows; i++) {
        for (usize j = 0; j < mat->cols; j++) {
            matrix_set(result, j, i, matrix_get(mat, i, j));
        }
    }
}

void matrix_add_row(Matrix *result, const Matrix *mat, const Matrix *row) {
    assert(result);
    assert(mat);
    assert(row);
    assert(mat->data);
    assert(row->data);
    assert(row->rows == 1);
    assert(mat->cols == row->cols);
    assert(result->rows == mat->rows);
    assert(result->cols == mat->cols);

    for (usize r = 0; r < mat->rows; r++) {
        usize offset = r * mat->cols;
        for (usize c = 0; c < mat->cols; c++) {
            result->data[offset + c] = mat->data[offset + c] + row->data[c];
        }
    }
}

void matrix_sum_rows(Matrix *result, const Matrix *mat) {
    for (usize r = 0; r < mat->rows; r++) {
        for (usize c = 0; c < mat->cols; c++) {
            result->data[c] += mat->data[r * mat->cols + c];
        }
    }
}

/*==============================================================================
 * In-place Arithmetic
 *============================================================================*/

void matrix_add_inplace(Matrix *a, const Matrix *b) {
    assert(a);
    assert(b);
    assert(a->rows == b->rows);
    assert(a->cols == b->cols);

    usize n = matrix_size(a);
    for (usize i = 0; i < n; i++) {
        a->data[i] += b->data[i];
    }
}

void matrix_subtract_inplace(Matrix *a, const Matrix *b) {
    assert(a);
    assert(b);
    assert(a->rows == b->rows);
    assert(a->cols == b->cols);

    usize n = matrix_size(a);
    for (usize i = 0; i < n; i++) {
        a->data[i] -= b->data[i];
    }
}

void matrix_scalar_multiply_inplace(Matrix *mat, float scalar) {
    assert(mat);
    assert(mat->data);

    usize n = matrix_size(mat);
    for (usize i = 0; i < n; i++) {
        mat->data[i] *= scalar;
    }
}

void matrix_add_row_inplace(Matrix *mat, const Matrix *row) {
    assert(mat);
    assert(row);
    assert(mat->data);
    assert(row->data);
    assert(row->rows == 1);
    assert(mat->cols == row->cols);

    for (usize r = 0; r < mat->rows; r++) {
        usize offset = r * mat->cols;
        for (usize c = 0; c < mat->cols; c++) {
            mat->data[offset + c] += row->data[c];
        }
    }
}

/*==============================================================================
 * Element-wise Operations
 *============================================================================*/

void matrix_apply_function(Matrix *mat, float (*func)(float)) {
    assert(mat);
    assert(mat->data);
    assert(func);

    usize n = matrix_size(mat);
    for (usize i = 0; i < n; i++) {
        mat->data[i] = func(mat->data[i]);
    }
}
