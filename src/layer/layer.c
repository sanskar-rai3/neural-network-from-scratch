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
#include "layer/layer.h"
#include "layer/dense.h"
#include "math/matrix.h"
#include "error/error.h"

/*==============================================================================
 * Creation & Destruction
 *============================================================================*/

Error layer_init(Layer *layer, const LayerConfig *config) {
    if (!layer || !config)
        return ERROR_INVALID_ARGS;

    switch (config->layer_type) {
        case LAYER_DENSE: {
            layer->layer_type = LAYER_DENSE;

            if (!dense_init(&layer->dense, &config->dense_config))
                return ERROR_ALLOCATION_FAILED;

            layer->activation = config->activation;

            break;
        }

        default:
            return ERROR_LAYER_UNKNOWN; 
    }

    layer->z_cache = matrix_empty();

    return ERROR_OK;
}

void layer_destroy(Layer *layer) {
    dense_destroy(&layer->dense);
    matrix_destroy(&layer->z_cache);

    layer->activation = ACT_NONE;
}

/*==============================================================================
 * Forward Pass
 *============================================================================*/

Error layer_forward(Matrix *output, Layer *layer, const Matrix *input) {
    if (!output || !layer || !input)
        return ERROR_INVALID_ARGS;

    if (!(input->cols == layer->dense.weights.rows) ||
        !(output->rows == input->rows)              ||
        !(output->cols == layer->dense.weights.cols)) {

        return ERROR_MATRIX_INVALID_SIZE;
    }

    switch (layer->layer_type) {
        case LAYER_DENSE: {
            /* Z = XW + b */
            dense_forward(output, &layer->dense, input);

            /* Cache Z for the backward pass */
            if (layer->z_cache.rows != output->rows ||
                layer->z_cache.cols != output->cols ||
               !layer->z_cache.data) {

                matrix_destroy(&layer->z_cache);

                if (matrix_create(&layer->z_cache, output->rows, output->cols))
                    return ERROR_ALLOCATION_FAILED;
            }

            matrix_copy(&layer->z_cache, output);

            /* A = activation(Z) */
            activation_forward(output, layer->activation);

            break;
        }

        default:
            return ERROR_LAYER_UNKNOWN;
    }

    return ERROR_OK;
}

/*==============================================================================
 *  Backward Pass
 *============================================================================*/

Error layer_backward(Matrix *d_input, Layer *layer, const Matrix *d_output) {
    if (!d_input || !layer || !d_output)
        return ERROR_INVALID_ARGS;

    if (!(d_output->rows == layer->z_cache.rows) ||
        !(d_output->cols == layer->z_cache.cols)) {

        return ERROR_MATRIX_INVALID_SIZE;
    }

    Matrix dZ;
    if (!matrix_create(&dZ, layer->z_cache.rows, layer->z_cache.cols))
        return ERROR_ALLOCATION_FAILED;

    /*
     * Activation backward:
     *
     * dZ = dA ⊙ f'(Z)
     *
     * dA = d_output
     */
    activation_backward(&dZ, d_output, &layer->z_cache, layer->activation);

    /*
     * Dense backward:
     *
     * dX = dZ * W^T
     * dW = X^T * dZ
     * db = sum(dZ)
     */
    Error outcome = dense_backward(d_input, &layer->dense, &dZ);
    if (!outcome)
        return outcome;

    matrix_destroy(&dZ);
}
