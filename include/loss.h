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

#ifndef LOSS_H
#define LOSS_H

#include "common.h"
#include "math/matrix.h"

/*==============================================================================
 * Forward pass
 *=============================================================================*/

/**
 * @brief Computes the Mean Squared Error (MSE) loss between predictions and targets.
 *
 * Formula: MSE = (1 / N) * sum((prediction - target)^2)
 *
 * @param prediction Pointer to the predicted output matrix (batch_size x outputs).
 * @param target     Pointer to the target ground truth matrix (batch_size x outputs).
 *
 * @return           The calculated MSE loss value.
 */
float loss_mse(const Matrix *prediction, const Matrix *target);

/**
 * @brief Computes the Categorical Cross-Entropy loss between predictions and targets.
 *
 * Formula: CE = -(1 / N) * sum(target * log(prediction + epsilon))
 *
 * @param prediction Pointer to predicted probability distribution matrix (batch_size x classes).
 * @param target     Pointer to one-hot encoded ground truth matrix (batch_size x classes).
 *
 * @return           The calculated Cross-Entropy loss value.
 */
float loss_cross_entropy(const Matrix *prediction, const Matrix *target);

/*==============================================================================
 * Loss Backward Pass
 *============================================================================*/

/**
 * @brief Computes the gradient of the Mean Squared Error loss with respect
 *        to the predictions.
 *
 * Formula:
 *     dL/dPrediction = (2 / N) * (prediction - target)
 *
 * @param d_prediction Destination matrix for the loss gradient.
 *                     Must have the same shape as prediction.
 * @param prediction   Pointer to the predicted output matrix.
 * @param target       Pointer to the target ground truth matrix.
 */
void loss_mse_backward(Matrix *d_prediction, const Matrix *prediction, const Matrix *target);

/**
 * @brief Computes the gradient of the Cross-Entropy loss with respect
 *        to the predictions.
 *
 * Formula:
 *     dL/dPrediction =
 *         -(1 / N) * target / prediction
 *
 * @param d_prediction Destination matrix for the loss gradient.
 *                     Must have the same shape as prediction.
 * @param prediction   Pointer to predicted probability distribution matrix.
 * @param target        Pointer to one-hot encoded ground truth matrix.
 */
void loss_cross_entropy_backward(Matrix *d_prediction, const Matrix *prediction, const Matrix *target);

/**
 * @brief Computes the gradient of the combined Softmax + Cross-Entropy loss
 *        with respect to the logits before the Softmax activation.
 *
 * Combining Softmax and Cross-Entropy simplifies the gradient to:
 *
 *     dL/dZ = (prediction - target) / batch_size
 *
 * where prediction is the Softmax output and Z is the input to Softmax.
 *
 * This fused form is numerically and computationally preferable to
 * separately computing the Softmax and Cross-Entropy derivatives.
 *
 * @param dZ         Destination matrix for the gradient with respect
 *                   to the Softmax input (logits).
 * @param prediction Pointer to the Softmax output matrix.
 * @param target     Pointer to one-hot encoded ground truth matrix.
 */
void loss_cross_entropy_softmax_backward( Matrix *dZ, const Matrix *prediction, const Matrix *target);

#endif /* LOSS_H */
