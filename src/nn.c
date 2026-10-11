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
#include "nn.h" 
#include "error/error.h"
#include "layer/layer.h"
#include "loss.h"
#include "math/matrix.h"
#include "error/error.h"

#include <stdio.h>
#include <stdlib.h>

/*==============================================================================
 * Creation & Destruction
 *============================================================================*/

Outcome network_init(Network *network, const LayerConfig *configs, usize layer_count) {
    if (!network || !configs || !(layer_count > 0))
        return OUTCOME_INVALID_ARGS;

    network->layers = malloc(layer_count * sizeof(Layer));
    if (!network->layers) {
        return OUTCOME_ALLOCATION_FAILED;
    }

    network->layer_count = layer_count;

    for (usize i = 0; i < layer_count; i++) {
        Outcome outcome = layer_init(&network->layers[i], &configs[i]);
        if (!outcome) {
            while (i > 0) {
                i--;
                layer_destroy(&network->layers[i]);
            }
            
            free(network->layers);
            network->layers = NULL;
            network->layer_count = 0;

            return outcome;
        }
    }

    return OUTCOME_OK;
}

void network_destroy(Network *network) {
    if (network->layers != NULL) {
        for (usize i = 0; i < network->layer_count; i++) {
            layer_destroy(&network->layers[i]);
        }

        free(network->layers);
        network->layers = NULL;
    }

    network->layer_count = 0;
}

/*==============================================================================
 * Forward Pass
 *============================================================================*/

Outcome network_forward(Matrix *output, const Network *network, const Matrix *input) {
    if (!output || !network || !input)
        return OUTCOME_INVALID_ARGS;

    /* Layer 1: Run input directly into a temp matrix */
    Matrix current;
    if (!matrix_create(&current, input->rows, network->layers[0].dense.weights.cols))
        return OUTCOME_ALLOCATION_FAILED;

    Outcome outcome = layer_forward(&current, &network->layers[0], input);
    if (!outcome)
        return outcome;

    /* Middle Layers: Run current -> next, swap */
    for (usize i = 1; i < network->layer_count - 1; i++) {
        Matrix next;
        if (!matrix_create(&next, input->rows, network->layers[i].dense.weights.cols))
            return OUTCOME_ALLOCATION_FAILED;

        outcome = layer_forward(&next, &network->layers[i], &current);
        if (!outcome)
            return outcome;

        matrix_destroy(&current);
        current = next;
    }

    /* Final Layer: Compute directly into output */
    if (network->layer_count > 1) {
        outcome = layer_forward(output, &network->layers[network->layer_count - 1], &current);
        if (!outcome)
            return outcome;

        matrix_destroy(&current);
    } else {
        /* If 1 layer total, output gets current's data */
        *output = current;
    }

    return OUTCOME_OK;
}

/*==============================================================================
 * Backward Pass
 *============================================================================*/

Outcome network_backward(Network *network, const Matrix *d_output) {
    if (!network || !d_output)
        return OUTCOME_INVALID_ARGS;

    /*
     * Initially:
     *
     * d_current = dL/dA of the final layer
     */
    Matrix d_current;
    if (!matrix_create(&d_current, d_output->rows, d_output->cols))
        return OUTCOME_ALLOCATION_FAILED;

    matrix_copy(&d_current, d_output);

    for (usize i = network->layer_count; i-- > 0;) {
        Layer *layer = &network->layers[i];

        /*
         * d_next = dL/dX for this layer.
         *
         * X has the same shape as the layer's cached input.
         */
        Matrix d_next;
        if (!matrix_create(&d_next, layer->dense.input_cache.rows, layer->dense.input_cache.cols))
            return OUTCOME_ALLOCATION_FAILED;

        /*
         * d_current = dL/dA
         * d_next    = dL/dX
         */
        Outcome outcome = layer_backward(&d_next, layer, &d_current);
        if (!outcome)
            return outcome;

        matrix_destroy(&d_current);

        d_current = d_next;
    }

    /*
     * d_current is now dL/dX for the entire network.
     */
    matrix_destroy(&d_current);

    return OUTCOME_OK;
}

/*==============================================================================
 * Saving and Loading Models 
 =============================================================================*/

Outcome network_save(const Network *network, const char *file_name) {
    FILE *file = fopen(file_name, "wb");
    if (!file) {
        return OUTCOME_FILE_CREATE_FAILED;
    }

    /* Saving the magic number */
    const u32 magic_num = NN_MAGIC;
    fwrite(&magic_num, sizeof(magic_num), 1, file);

    fwrite(&network->epoch_count, sizeof(usize), 1, file);
    fwrite(&network->layer_count, sizeof(usize), 1, file);

    for (usize i = 0; i < network->layer_count; i++) {
        Layer *layer = &network->layers[i];

        /* Save the layer type and layer's activation function */
        fwrite(&layer->layer_type, sizeof(layer->layer_type), 1, file);
        fwrite(&layer->activation, sizeof(layer->activation), 1, file);

        switch (layer->layer_type) {
            case LAYER_DENSE: {
                Dense *dense = &layer->dense;

                /* Save matrix dimensions */
                fwrite(&dense->weights.rows, sizeof(usize), 1, file);
                fwrite(&dense->weights.cols, sizeof(usize), 1, file);

                /* Save matrix data */
                usize weight_count = dense->weights.rows * dense->weights.cols;

                fwrite(dense->weights.data, sizeof(float), weight_count, file);

                /* Bias matrix */
                fwrite(&dense->bias.rows, sizeof(usize), 1, file);
                fwrite(&dense->bias.cols, sizeof(usize), 1, file);

                usize bias_count = dense->bias.rows * dense->bias.cols;

                fwrite(dense->bias.data, sizeof(float), bias_count, file);

                break;
            }

            default:
                return OUTCOME_LAYER_UNKNOWN;
        }
    }

    return OUTCOME_OK;
}


Outcome network_load(Network *network, const char *file_name) {
    FILE *file = fopen(file_name, "rb");
    if (!file) {
        return OUTCOME_FILE_OPEN_FAILED;
    }

    u32 magic_num;
    fread(&magic_num, sizeof(u32), 1, file);
    if (magic_num != NN_MAGIC) {
        return OUTCOME_FILE_INVALID;
    }

    fread(&network->epoch_count, sizeof(usize), 1, file);
    fread(&network->layer_count, sizeof(usize), 1, file);

    network->layers = malloc(network->layer_count * sizeof(*network->layers));

    for (usize i = 0; i < network->layer_count; i++) {
        Layer *layer = &network->layers[i];

        /* Load the layer type and layer's activation */
        fread(&layer->layer_type, sizeof(layer->layer_type), 1, file);
        fread(&layer->activation, sizeof(layer->activation), 1, file);

        layer->z_cache = matrix_empty();

        switch (layer->layer_type) {
            case LAYER_DENSE: {
                Dense *dense = &layer->dense;

                /* Load matrix dimensions */
                fread(&dense->weights.rows, sizeof(usize), 1, file);
                fread(&dense->weights.cols, sizeof(usize), 1, file);

                /* Load matrix data */
                usize weight_count = dense->weights.rows * dense->weights.cols;

                dense->weights.data = malloc(weight_count * sizeof(float));
                fread(dense->weights.data, sizeof(float), weight_count, file);

                /* Bias matrix */
                fread(&dense->bias.rows, sizeof(usize), 1, file);
                fread(&dense->bias.cols, sizeof(usize), 1, file);

                usize bias_count = dense->bias.rows * dense->bias.cols;

                dense->bias.data = malloc(bias_count * sizeof(float));
                fread(dense->bias.data, sizeof(float), bias_count, file);

                dense->input_cache = matrix_empty();

                matrix_create(&dense->d_weights, dense->weights.rows, dense->weights.cols);
                matrix_create(&dense->d_bias, dense->bias.rows, dense->bias.cols);

                break;
            }

            default:
                return OUTCOME_LAYER_UNKNOWN;
        }
    }

    return OUTCOME_OK;
}

/*==============================================================================
 * Training
 *============================================================================*/

static void labels_to_one_hot(Matrix *target, const u8 *labels) {
    matrix_fill(target, 0.0f);

    for (usize i = 0; i < target->rows; i++) {
        target->data[i * target->cols + labels[i]] = 1.0f;
    }
}

void network_train(Network *nn, const Dataset *dataset, Optimizer *optimizer, usize epochs, usize batch_size){
    if (!nn || !dataset || !optimizer ||
        !dataset->data || !dataset->label ||
        dataset->size == 0 ||
        batch_size == 0 ||
        batch_size > dataset->size) {
        
        return;
    }

    Matrix X;
    Matrix Y;
    Matrix prediction;
    Matrix d_output;

    if (!matrix_create(&X, batch_size, dataset->input_count) ||
        !matrix_create(&Y, batch_size, dataset->class_count) ||
        !matrix_create(&prediction, batch_size, dataset->class_count) ||
        !matrix_create(&d_output, batch_size, dataset->class_count)) {

        goto cleanup;
    }

    const usize batch_count = dataset->size / batch_size;

    for (usize epoch = 0; epoch < epochs; epoch++) {
        float epoch_loss = 0.0f;

        for (usize batch = 0; batch < batch_count; batch++) {
            const usize start = batch * batch_size;

            /* Copy input batch */
            for (usize i = 0; i < batch_size; i++) {
                for (usize j = 0; j < dataset->input_count; j++) {
                    X.data[i * dataset->input_count + j] =
                        dataset->data[
                            (start + i) * dataset->input_count + j
                        ];
                }
            }

            /* Convert labels to one-hot vectors */
            labels_to_one_hot(&Y, dataset->label + start);

            /* Forward pass */
            network_forward(&prediction, nn, &X);

            /* Calculate loss */
            epoch_loss += loss_cross_entropy(&prediction, &Y);

            /* Softmax + cross-entropy backward */
            loss_cross_entropy_softmax_backward(
                &d_output, &prediction, &Y);

            /* Backpropagation */
            network_backward(nn, &d_output);

            /* Update parameters */
            optimizer_step(optimizer, nn);
        }

        // epoch_loss /= (float)batch_count;

        // printf("epoch: %2zu | loss: %.6f\n",
        //        epoch + 1, epoch_loss);
    }

    matrix_destroy(&X);
    matrix_destroy(&Y);
    matrix_destroy(&prediction);
    matrix_destroy(&d_output);

    return;

cleanup:
    matrix_destroy(&X);
    matrix_destroy(&Y);
    matrix_destroy(&prediction);
    matrix_destroy(&d_output);

    return;
}

static int argmax(const Matrix *prediction) {
    int best = 0;

    for (int i = 1; i < (int)prediction->cols; i++) {
        if (prediction->data[i] > prediction->data[best]) {
            best = i;
        }
    }

    return best;
}

void network_evaluate(float *ret_accuracy, const Network *nn, const Dataset *dataset) {
    if (!nn || !dataset || !ret_accuracy ||
        !dataset->data || !dataset->label ||
        dataset->size == 0 ||
        dataset->input_count == 0 ||
        dataset->class_count == 0) {

        return;
    }

    /* Creating prediction matrices */
    Matrix input = {0};
    Matrix prediction = {0};

    if (!matrix_create(&input, 1, dataset->input_count) ||
        !matrix_create(&prediction, 1, dataset->class_count)) {

        matrix_destroy(&input);
        matrix_destroy(&prediction);

        return;
    }

    /* Testing model accuracy */
    usize correct = 0;

    for (usize i = 0; i < dataset->size; i++) {
        /* Copy one image into input */
        for (usize j = 0; j < dataset->input_count; j++) {
            input.data[j] =
                dataset->data[i * dataset->input_count + j];
        }

        /* Forward pass */
        network_forward(&prediction, nn, &input);

        /* Get predicted class */
        usize predicted = (usize)argmax(&prediction);

        /* Compare */
        if (predicted == dataset->label[i]) {
            correct++;
        }
    }

    *ret_accuracy = (float)correct / (float)dataset->size * 100.0f;

    matrix_destroy(&input);
    matrix_destroy(&prediction);
}