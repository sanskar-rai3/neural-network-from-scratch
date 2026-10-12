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

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "common.h"
#include "error/error.h"

/* Forward declaration to avoid including nn.h here. */
typedef struct Network Network;

/*==============================================================================
 * Types
 *============================================================================*/

/**
 * @brief Specifies the optimization algorithm used to update parameters.
 */
typedef enum OptimizerType {
    OPT_NULL = 0,  /**< No optimization algorithm. */
    OPT_SGD        /**< Stochastic Gradient Descent. */
} OptimizerType;

/**
 * @brief Configuration for a neural network optimizer.
 */
typedef struct Optimizer {
    OptimizerType type;  /**< Optimization algorithm to use. */
    float learning_rate; /**< Step size used for parameter updates. */
} Optimizer;

/*==============================================================================
 * Creation & Destruction
 *============================================================================*/

/**
 * @brief Initializes an optimizer.
 *
 * @param optimizer    Pointer to the Optimizer to initialize.
 * @param learning_rate Learning rate used during parameter updates.
 * @param type          Optimization algorithm to use.
 */
Error optimizer_init(Optimizer *optimizer, float learning_rate, OptimizerType type);

/**
 * @brief Destroys an optimizer and releases any resources it owns.
 *
 * @param optimizer Pointer to the Optimizer to destroy.
 */
void optimizer_destroy(Optimizer *optimizer);

/*==============================================================================
 * Optimization
 *============================================================================*/

/**
 * @brief Updates all trainable parameters in the network using their gradients.
 *
 * For SGD, parameters are updated according to:
 *
 *     W = W - learning_rate * dW
 *     b = b - learning_rate * db
 *
 * @param optimizer Pointer to the initialized Optimizer.
 * @param network   Pointer to the Network whose parameters will be updated.
 */
Error optimizer_step(Optimizer *optimizer, Network *network);

#endif /* OPTIMIZER_H */
