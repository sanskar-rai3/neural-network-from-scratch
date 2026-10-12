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
#include "layer/dense.h"
#include "error/error.h"
#include "math/matrix.h"

/*==============================================================================
 * Creation & Destruction
 *============================================================================*/

Error dense_init(Dense *layer, const DenseConfig *config) {
    Error err;
    if (!layer || !config)
        return ERROR_INVALID_ARGS;

    err = matrix_create(&layer->weights, config->input_size, config->output_size);
    if (err != ERROR_OK)
        return err;

    err = matrix_create(&layer->bias, 1, config->output_size);
    if (err != ERROR_OK)
        return err;

    err = matrix_create(&layer->d_weights, config->input_size, config->output_size);
    if (err != ERROR_OK)
        return err;

    err = matrix_create(&layer->d_bias, 1, config->output_size);
    if (err != ERROR_OK)
        return err;

    /* Randomizing weights with He initialization */
    matrix_he_uniform(&layer->weights, config->input_size);

    /* Initializing bias matrix with 0 */
    matrix_fill(&layer->bias, 0.0f);

    layer->input_cache = matrix_empty();

    return ERROR_OK;
}

void dense_destroy(Dense *layer) {
    matrix_destroy(&layer->weights);
    matrix_destroy(&layer->bias);

    matrix_destroy(&layer->d_weights);
    matrix_destroy(&layer->d_bias);

    matrix_destroy(&layer->input_cache);
}
/*==============================================================================
 * Forward Pass
 *============================================================================*/

Error dense_forward(Matrix *output, Dense *layer, const Matrix *input) {
    if (!output || !layer || !input)
        return ERROR_INVALID_ARGS;

    if (input->cols != layer->weights.rows ||
        output->rows != input->rows ||
        output->cols != layer->weights.cols) {

        return ERROR_MATRIX_INVALID_SIZE;
    }

    /* Cache X for backward propagation */
    if (layer->input_cache.rows != input->rows ||
        layer->input_cache.cols != input->cols ||
       !layer->input_cache.data) {
        
        matrix_destroy(&layer->input_cache);

        Error err = matrix_create(&layer->input_cache, input->rows, input->cols);
        if (err != ERROR_OK)
            return err;
    }

    matrix_copy(&layer->input_cache, input);

    /* Z = XW + b */
    matrix_multiply(output, input, &layer->weights);
    matrix_add_row_inplace(output, &layer->bias);

    return ERROR_OK;
}

/*==============================================================================
 * Backward Pass
 *============================================================================*/

Error dense_backward(Matrix *d_input, Dense *layer, const Matrix *dZ) {
    Error err;
    if (!d_input || !layer || !dZ)
        return ERROR_INVALID_ARGS;

    const Matrix *X = &layer->input_cache;
    const Matrix *W = &layer->weights;

    if (!(dZ->rows == X->rows) || !(dZ->cols == W->cols))
        return ERROR_MATRIX_INVALID_SIZE;

    /*
     * dW = X^T * dZ
     */
    Matrix X_T;
    err = matrix_create(&X_T, X->cols, X->rows);
    if (err != ERROR_OK)
        return err;

    matrix_transpose(&X_T, X);
    matrix_multiply(&layer->d_weights, &X_T, dZ);

    matrix_destroy(&X_T);

    /*
     * db = sum(dZ over the batch)
     */
    matrix_fill(&layer->d_bias, 0.0f);
    matrix_sum_rows(&layer->d_bias, dZ);

    /*
     * dX = dZ * W^T
     */
    Matrix W_T;
    err = matrix_create(&W_T, W->cols, W->rows);
    if (err != ERROR_OK)
        return err;

    matrix_transpose(&W_T, W);
    matrix_multiply(d_input, dZ, &W_T);

    matrix_destroy(&W_T);

    return ERROR_OK;
}
