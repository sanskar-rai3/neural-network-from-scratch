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

#ifndef NETWORK_H
#define NETWORK_H

#include "common.h"
#include "layer/layer.h"
#include "math/matrix.h"
#include "data-loader/dataset.h"
#include "error/error.h"
#include "optimizer.h"

/*==============================================================================
 * Network Structure Definition
 *============================================================================*/

/**
 * @brief Sequential neural network model composed of stacked layers.
 */
typedef struct Network {
    Layer *layers;      /**< Dynamic array of sequential Layer structures */
    usize  layer_count; /**< Total number of layers in the network */
    usize  epoch_count; /**< Total number of epochs trained */
} Network;


/*==============================================================================
 * Creation & Destruction
 *============================================================================*/

/**
 * @brief Allocates and initializes a sequential neural network from layer configurations.
 *
 * @param network     Pointer to the Network structure to initialize.
 * @param configs     Array of LayerConfig structures defining each layer.
 * @param layer_count Number of layers to create.
 *
 * @return ERROR_OK on success, corresponding error code on failure.
 */
Error network_init(Network *network, const LayerConfig *configs, usize layer_count);

/**
 * @brief Frees all dynamic memory owned by the network and its constituent layers.
 *
 * @param network Pointer to the Network to destroy.
 */
void network_destroy(Network *network);


/*==============================================================================
 * Forward Pass
 *============================================================================*/

/**
 * @brief Performs a forward pass through all layers sequentially.
 *
 * Computes: output = Layer_N(... Layer_2(Layer_1(input)))
 *
 * @param output Destination matrix to store the final output (batch_size x final_output_size).
 * @param network Pointer to the initialized Network.
 * @param input   Input matrix (batch_size x initial_input_size).
 *
 * @return ERROR_OK on success, corresponding error code on failure.
 */
Error network_forward(Matrix *output, const Network *network, const Matrix *input);

/*==============================================================================
 * Backward Pass
 *============================================================================*/

/**
 * @brief Performs a backward pass through all layers in reverse order.
 *
 * Propagates the output gradient backward through every layer, computing
 * gradients for each layer's weights and biases.
 *
 * @param network  Pointer to the initialized Network.
 * @param d_output Gradient of the loss with respect to the network output
 *                 (dL/dA), with shape (batch_size x output_size).
 *
 * @return ERROR_OK on success, corresponding error code on failure.
 */
Error network_backward(Network *network, const Matrix *d_output);

/*==============================================================================
 * Saving and Loading Models 
 *============================================================================*/

/**
 * @brief Magic number for the network data binary.
 */
#define NN_MAGIC 0x132B5FAu

/**
 * @brief Saves all the parameter of a given network.
 *
 * @param network Pointer to the input network.
 * @param file_name Name of the file to store the data.
 *
 * @return ERROR_OK on success, corresponding error code on failure.
 */
Error network_save(const Network *network, const char *file_name);

/**
 * @brief Loads all the data from file.
 *
 * @param network Pointer to the output network.
 * @param file_name Name of the file to load the data from.
 *
 * @return ERROR_OK on success, corresponding error code on failure.
 */
Error network_load(Network *network, const char *file_name);

/*==============================================================================
 * Training
 *============================================================================*/

/**
 * @brief Trains network for specified epoch.
 * 
 * @param network Pointer to the network to train.
 * @param dataset Pointer to the dataset.
 * @param epoch Number of epochs to train.
 *
 * @return ERROR_OK on success, corresponding error code on failure.
 */
Error network_train(Network *network, const Dataset *dataset, Optimizer *optimizer, usize epoch, usize batch_size);

/**
 * @brief Evaluates the accuracy of a Network model.
 * 
 * @param ret_accuracy Pointer to the float to store accuracy.
 * @param network Pointer to the network to evaluate.
 * @param dataset Pointer to the dataset.
 *
 * @return ERROR_OK on success, corresponding error code on failure.
 */
Error network_evaluate(float *ret_accuracy, const Network *network, const Dataset *dataset);

#endif /* NETWORK_H */
