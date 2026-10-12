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

#ifndef MATRIX_H
#define MATRIX_H

#include "common.h"
#include "error/error.h"

/*==============================================================================
 * Matrix Structure Definition
 *============================================================================*/

/**
 * @brief 2D Dense Matrix structure stored in row-major order.
 */
typedef struct Matrix {
    usize  rows;  /**< Number of matrix rows */
    usize  cols;  /**< Number of matrix columns */
    float *data;  /**< Pointer to contiguous 1D array of float elements (size: rows * cols) */
} Matrix;

/*==============================================================================
 * Creation & Destruction
 *============================================================================*/

/**
 * @brief Allocates memory for a matrix of the given size.
 *
 * @param mat  Pointer to the Matrix structure to initialize.
 * @param rows Number of rows (must be > 0).
 * @param cols Number of columns (must be > 0).
 *
 * @return 1 on success, 0 on failure.
 */
Error matrix_create(Matrix *mat, usize rows, usize cols);

/**
 * @brief Creates a matrix by copying data from an existing buffer.
 *
 * @param mat    Pointer to the Matrix structure to initialize.
 * @param rows   Number of rows (must be > 0).
 * @param cols   Number of columns (must be > 0).
 * @param buffer Source buffer containing (rows * cols) float elements.
 *
 * @return 1 on success, 0 on failure.
 */
Error matrix_copy_buffer(Matrix *mat, usize rows, usize cols, float *buffer);

/**
 * @brief Releases the memory owned by a matrix and resets its dimensions.
 *
 * @param mat Pointer to the Matrix structure to destroy.
 */
void matrix_destroy(Matrix *mat);


/*==============================================================================
 * Element Access
 *============================================================================*/

/**
 * @brief Returns the value at the specified row and column.
 *
 * @param mat Pointer to the matrix.
 * @param row Zero-based row index.
 * @param col Zero-based column index.
 * @return    The float value stored at (row, col).
 */
float matrix_get(const Matrix *mat, usize row, usize col);

/**
 * @brief Sets the value at the specified row and column.
 *
 * @param mat   Pointer to the matrix.
 * @param row   Zero-based row index.
 * @param col   Zero-based column index.
 * @param value The float value to set.
 */
void matrix_set(Matrix *mat, usize row, usize col, float value);


/*==============================================================================
 * Utility
 *============================================================================*/

/**
 * @brief Returns the total number of elements in the matrix (rows * cols).
 *
 * @param mat Pointer to the matrix.
 * @return    Total element count.
 */
usize matrix_size(const Matrix *mat);

/**
 * @brief Returns an unallocated matrix structure with zero dimensions.
 *
 * @return Empty Matrix structure.
 */
Matrix matrix_empty(void);

/**
 * @brief Fills every element of a matrix with a scalar value.
 *
 * @param mat   Pointer to the matrix to fill.
 * @param value The float value to fill with.
 */
void matrix_fill(Matrix *mat, float value);

/**
 * @brief Copies all elements from a source matrix into a destination matrix.
 *
 * Both matrices must have identical dimensions.
 *
 * @param dest Pointer to the destination matrix.
 * @param src  Pointer to the source matrix.
 */
void matrix_copy(Matrix *dest, const Matrix *src);


/*==============================================================================
 * Random Initialization
 *============================================================================*/

/**
 * @brief Fills the matrix with uniformly distributed random values in [min, max].
 *
 * @param mat Pointer to the matrix to randomize.
 * @param min Minimum bound.
 * @param max Maximum bound.
 */
void matrix_randomize(Matrix *mat, float min, float max);

/**
 * @brief Initializes the matrix using He uniform initialization.
 *
 * Recommended for ReLU activation networks.
 *
 * @param mat    Pointer to the matrix.
 * @param fan_in Number of input units.
 */
void matrix_he_uniform(Matrix *mat, usize fan_in);

/**
 * @brief Initializes the matrix using Xavier/Glorot uniform initialization.
 *
 * Recommended for tanh or sigmoid activation networks.
 *
 * @param mat     Pointer to the matrix.
 * @param fan_in  Number of input units.
 * @param fan_out Number of output units.
 */
void matrix_xavier_uniform(Matrix *mat, usize fan_in, usize fan_out);


/*==============================================================================
 * Matrix Arithmetic
 *============================================================================*/

/**
 * @brief Computes element-wise addition: result = a + b.
 *
 * @param result Pointer to the destination matrix.
 * @param a      Pointer to the first operand matrix.
 * @param b      Pointer to the second operand matrix.
 */
void matrix_add(Matrix *result, const Matrix *a, const Matrix *b);

/**
 * @brief Computes element-wise subtraction: result = a - b.
 *
 * @param result Pointer to the destination matrix.
 * @param a      Pointer to the first operand matrix.
 * @param b      Pointer to the second operand matrix.
 */
void matrix_subtract(Matrix *result, const Matrix *a, const Matrix *b);

/**
 * @brief Computes matrix multiplication: result = a * b.
 *
 * result dimensions must be (a->rows x b->cols).
 *
 * @param result Pointer to the destination matrix.
 * @param a      Pointer to the left matrix.
 * @param b      Pointer to the right matrix.
 */
void matrix_multiply(Matrix *result, const Matrix *a, const Matrix *b);

/**
 * @brief Computes Hadamard (element-wise) product: result = a ⊙ b.
 *
 * @param result Pointer to the destination matrix.
 * @param a      Pointer to the first matrix.
 * @param b      Pointer to the second matrix.
 */
void matrix_hadamard(Matrix *result, const Matrix *a, const Matrix *b);

/**
 * @brief Computes scalar multiplication: result = mat * scalar.
 *
 * @param result Pointer to the destination matrix.
 * @param mat    Pointer to the source matrix.
 * @param scalar Scalar value to multiply by.
 */
void matrix_scalar_multiply(Matrix *result, const Matrix *mat, float scalar);

/**
 * @brief Computes the matrix transpose: result = mat^T.
 *
 * result dimensions must be (mat->cols x mat->rows).
 *
 * @param result Pointer to the destination matrix.
 * @param mat    Pointer to the matrix to transpose.
 */
void matrix_transpose(Matrix *result, const Matrix *mat);

/**
 * @brief Adds a row vector to every row of a matrix (broadcasting).
 *
 * @param result Pointer to the pre-allocated destination matrix (same dimensions as mat).
 * @param mat    Pointer to the input matrix (rows x cols).
 * @param row    Pointer to the row vector matrix (1 x cols).
 */
void matrix_add_row(Matrix *result, const Matrix *mat, const Matrix *row);

/**
 * @brief Adds the sum of each row of a matrix to the corresponding
 *        element of the destination vector.
 *
 * @param result Pointer to the destination matrix
 * @param mat Pointer to the source matrix
 */
void matrix_sum_rows(Matrix *result, const Matrix *mat);

/*==============================================================================
 * In-place Arithmetic
 *============================================================================*/

/**
 * @brief Performs in-place element-wise addition: a = a + b.
 *
 * @param a Pointer to the target matrix to update.
 * @param b Pointer to the matrix to add.
 */
void matrix_add_inplace(Matrix *a, const Matrix *b);

/**
 * @brief Performs in-place element-wise subtraction: a = a - b.
 *
 * @param a Pointer to the target matrix to update.
 * @param b Pointer to the matrix to subtract.
 */
void matrix_subtract_inplace(Matrix *a, const Matrix *b);

/**
 * @brief Performs in-place scalar multiplication: mat = mat * scalar.
 *
 * @param mat    Pointer to the matrix to update.
 * @param scalar Scalar value to multiply by.
 */
void matrix_scalar_multiply_inplace(Matrix *mat, float scalar);

/**
 * @brief Adds a row vector to every row of a matrix in-place (broadcasting).
 *
 * @param mat Pointer to the matrix to update in-place (rows x cols).
 * @param row Pointer to the row vector matrix (1 x cols).
 */
void matrix_add_row_inplace(Matrix *mat, const Matrix *row);


/*==============================================================================
 * Element-wise Operations
 *============================================================================*/

/**
 * @brief Applies a scalar function to every element of the matrix in-place.
 *
 * @param mat  Pointer to the matrix to update.
 * @param func Function pointer taking and returning a float.
 */
void matrix_apply_function(Matrix *mat, float (*func)(float));

#endif /* MATRIX_H */
