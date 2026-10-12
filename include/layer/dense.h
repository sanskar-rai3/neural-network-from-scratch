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

#ifndef DENSE_H
#define DENSE_H

#include "common.h"
#include "math/matrix.h"
#include "error/error.h"

/**
 * @brief Fully connected (Dense) layer structure containing weights and biases.
 */
typedef struct Dense {
    Matrix weights;     /**< Weight matrix: (input_size x output_size) */
    Matrix bias;        /**< Bias matrix: (1 x output_size) */

    /* Cached values from the forward pass */
    Matrix input_cache; /**< Input X: (batch_size x input_size) */

    /* Gradients computed during the backward pass */
    Matrix d_weights;   /**< Gradient dL/dW: same shape as weights */
    Matrix d_bias;      /**< Gradient dL/db: same shape as bias */
} Dense;

/**
 * @brief Configuration parameters for Dense Layer
 */
typedef struct DenseConfig {
    usize input_size;  /**< Number of input features */
    usize output_size; /**< Number of output features/neurons */
} DenseConfig;

/*==============================================================================
 * Creation & Destruction
 *============================================================================*/

/**
 * @brief Initializes a dense layer with allocated weights and biases.
 * 
 * Weights are typically initialized using He or Xavier uniform initialization,
 * and biases are zero-initialized.
 * 
 * @param layer       Pointer to the Dense layer structure to initialize.
 * @param config      Pointer to the configurations for the dense layer.
 *
 * @return            ERROR_OK on success, corresponding error code on failure.
 */
Error dense_init(Dense *layer, const DenseConfig *config);

/**
 * @brief Frees all memory associated with a dense layer's weights and biases.
 * 
 * @param layer Pointer to the Dense layer to destroy.
 */
void dense_destroy(Dense *layer);

/*==============================================================================
 *  Forward Pass
 *============================================================================*/

/**
 * @brief Performs a forward pass through the dense layer: output = (input * weights) + bias.
 * 
 * @param output Destination matrix to store the result. Must be pre-allocated 
 *               with shape (batch_size x output_size).
 * @param layer  Pointer to the initialized Dense layer.
 * @param input  Input matrix of shape (batch_size x input_size).
 *
 * @return            ERROR_OK on success, corresponding error code on failure.
 */
Error dense_forward(Matrix *output, Dense *layer, const Matrix *input);

/*==============================================================================
 *  Backward Pass
 *============================================================================*/

/**
 * @brief Performs a backward pass through the dense layer, computing the
 *        gradients with respect to the weights, bias, and input.
 *
 * @param d_input   Destination matrix for the gradient with respect to the
 *                  input (dL/dInput). Must be pre-allocated with shape
 *                  (batch_size x input_size).
 * @param d_weights Destination matrix for the gradient with respect to the
 *                  weights (dL/dWeights). Must be pre-allocated with the same
 *                  shape as the layer's weights.
 * @param d_bias    Destination matrix for the gradient with respect to the
 *                  bias (dL/dBias). Must be pre-allocated with the same shape
 *                  as the layer's bias.
 * @param layer     Pointer to the initialized Dense layer.
 * @param input     Input matrix from the forward pass, of shape
 *                  (batch_size x input_size).
 * @param d_output  Gradient with respect to the dense layer's output
 *                  (dL/dOutput), of shape (batch_size x output_size).
 *
 * @return            ERROR_OK on success, corresponding error code on failure.
 */
Error dense_backward(Matrix *d_input, Dense *layer, const Matrix *dZ);

#endif /* DENSE_H */
