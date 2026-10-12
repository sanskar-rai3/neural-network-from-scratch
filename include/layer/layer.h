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

#ifndef LAYER_H
#define LAYER_H

#include "common.h"
#include "layer/dense.h"
#include "activation.h"
#include "error/error.h"

/*==============================================================================
 * Types & Configuration
 *============================================================================*/

/**
 * @brief Enumeration of the different types of neural network layers.
 */
typedef enum LayerType {
    LAYER_DENSE = 0, /**< Fully connected dense layer. */
} LayerType;

/**
 * @brief Represents a neural network layer.
 *
 * A Layer contains the parameters and state required to perform
 * forward and backward propagation for a specific layer type.
 */
typedef struct Layer {
    Dense dense; /**< Dense layer parameters and state. */
    ActivationType activation; /**< Activation function applied to the layer output. */
    Matrix z_cache; /**< Cached pre-activation values used during backpropagation. */
} Layer;

/**
 * @brief Configuration parameters used to initialize a neural network layer.
 *
 * The configuration stored in the union depends on the specified layer type.
 */
typedef struct LayerConfig {
    DenseConfig dense_config; /**< Configuration for a dense layer. */

    ActivationType activation; /**< Activation function to apply to the layer output. */
} LayerConfig;


/*==============================================================================
 * Creation & Destruction
 *============================================================================*/

/**
 * @brief Initializes a Layer according to the specified configuration.
 *
 * @param layer  Pointer to the Layer structure to initialize.
 * @param config Pointer to the layer configuration parameters.
 * @return       1 on success, 0 on failure.
 */
Error layer_init(Layer *layer, const LayerConfig *config);

/**
 * @brief Frees all allocated memory owned by the layer.
 *
 * @param layer Pointer to the Layer to destroy.
 */
void layer_destroy(Layer *layer);


/*==============================================================================
 * Forward Pass
 *============================================================================*/

/**
 * @brief Performs a complete forward pass through the layer: output = activation(input * W + b).
 *
 * @param output Destination matrix to store the results (batch_size x output_size).
 * @param layer  Pointer to the initialized Layer.
 * @param input  Input matrix (batch_size x input_size).
 */
Error layer_forward(Matrix *output, Layer *layer, const Matrix *input);

/*==============================================================================
 *  Backward Pass
 *============================================================================*/

/**
 * @brief Performs a backward pass through the layer, propagating the gradient
 *        through the activation function and underlying dense layer.
 *
 * @param d_input   Destination matrix for the gradient with respect to the
 *                  layer input (dL/dInput). Must be pre-allocated with shape
 *                  (batch_size x input_size).
 * @param d_weights Destination matrix for the gradient with respect to the
 *                  layer weights (dL/dW). Must be pre-allocated with the same
 *                  shape as the layer's weights.
 * @param d_bias    Destination matrix for the gradient with respect to the
 *                  layer bias (dL/db). Must be pre-allocated with the same
 *                  shape as the layer's bias.
 * @param layer     Pointer to the initialized Layer.
 * @param d_output  Gradient with respect to the layer output (dL/dA).
 *                  Must have the same shape as the layer's cached Z.
 */
Error layer_backward(Matrix *d_input, Layer *layer, const Matrix *d_output);

#endif /* LAYER_H */
