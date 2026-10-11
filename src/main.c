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
#include "data-loader/dataset.h"
#include "optimizer.h"
#include "math/matrix.h"

#include "data-loader/emnist_loader.h"

#include <stdio.h>
#include <stdlib.h>

#define EPOCH         5
#define LEARNING_RATE 0.01

#define INPUT_SIZE  (EMNIST_ROWS * EMNIST_COLS)
#define HIDDEN1_SIZE 256
#define HIDDEN2_SIZE 128
#define OUTPUT_SIZE  10

#define BATCH_SIZE 1

#define TRAIN_IMAGE_PATH  "data/mnist/emnist-mnist-train-images-idx3-ubyte"
#define TRAIN_LABEL_PATH  "data/mnist/emnist-mnist-train-labels-idx1-ubyte"

#define TEST_IMAGE_PATH  "data/mnist/emnist-mnist-test-images-idx3-ubyte"
#define TEST_LABEL_PATH  "data/mnist/emnist-mnist-test-labels-idx1-ubyte"

int main(void) {
    /* Opening mnist train files */
    FILE *train_image = fopen(TRAIN_IMAGE_PATH, "rb");
    if (!train_image) {
        perror("train_image Error");
        return 1;
    }

    FILE *train_label = fopen(TRAIN_LABEL_PATH, "rb");
    if (!train_label) {
        perror("train_label Error");
        fclose(train_image);
        return 1;
    }

    /* Reading mnist train headers */
    EMNIST_IMAGE_HEADER train_image_header;
    EMNIST_LABEL_HEADER train_label_header;

    emnist_read_image_header(&train_image_header, train_image);
    emnist_read_label_header(&train_label_header, train_label);

    const usize TRAIN_IMAGE_COUNT = train_image_header.count;

    if (train_image_header.count != train_label_header.count) {
        fprintf(stderr, "Image/label count mismatch in training data\n");

        fclose(train_image);
        fclose(train_label);

        return 1;
    }

    Dataset dataset;
    dataset_create(&dataset, TRAIN_IMAGE_COUNT, INPUT_SIZE, OUTPUT_SIZE);

    /* Reading mnist train data */
    emnist_read_image_data_normalized(dataset.data, train_image, TRAIN_IMAGE_COUNT);
    emnist_read_label_data(dataset.label, train_label, TRAIN_IMAGE_COUNT);

    /* Closing mnist train files */
    fclose(train_image);
    fclose(train_label);

    /*=====================================================================*/

    /* Configuring layers */
    LayerConfig configs[] = {
        {
            .layer_type  = LAYER_DENSE,

            .dense_config.input_size  = INPUT_SIZE,
            .dense_config.output_size = HIDDEN1_SIZE,

            .activation = ACT_RELU
        },
        {
            .layer_type = LAYER_DENSE,

            .dense_config.input_size  = HIDDEN1_SIZE,
            .dense_config.output_size = HIDDEN2_SIZE,

            .activation = ACT_RELU
        },
        {
            .layer_type = LAYER_DENSE,

            .dense_config.input_size  = HIDDEN2_SIZE,
            .dense_config.output_size = OUTPUT_SIZE,

            .activation = ACT_SOFTMAX_CEL
        }
    };

    /* Network initialization */
    Network nn;
    if (!network_init(&nn, configs, sizeof(configs) / sizeof(configs[0]))) {
        fprintf(stderr, "Error initializing network\n");

        dataset_destroy(&dataset);
        return 1;
    }

    // network_load(&nn, "models/mnist-model-10epoch.nmf");

    /* Optimizer initialization */
    Optimizer optimizer;
    optimizer_init(&optimizer, LEARNING_RATE, OPT_SGD);

    /*=====================================================================*/

    /* Training */
    network_train(&nn, &dataset, &optimizer, 2, 1);

    // network_save(&nn, "models/mnist-model-15epoch.nmf");

    /*=====================================================================*/

    /* Opening mnist test files */
    FILE *test_image = fopen(TEST_IMAGE_PATH, "rb");
    if (!test_image) {
        perror("test_image");
        optimizer_destroy(&optimizer);
        network_destroy(&nn);
        dataset_destroy(&dataset);
        return 1;
    }

    FILE *test_label = fopen(TEST_LABEL_PATH, "rb");
    if (!test_label) {
        perror("test_label");
        fclose(test_image);

        optimizer_destroy(&optimizer);
        network_destroy(&nn);
        dataset_destroy(&dataset);

        return 1;
    }

    /* Reading mnist test headers */
    EMNIST_IMAGE_HEADER test_image_header;
    EMNIST_LABEL_HEADER test_label_header;

    emnist_read_image_header(&test_image_header, test_image);
    emnist_read_label_header(&test_label_header, test_label);

    const usize TEST_IMAGE_COUNT = test_image_header.count;

    if (test_image_header.count != test_label_header.count) {
        fprintf(stderr, "Test image/label count mismatch\n");

        fclose(test_image);
        fclose(test_label);

        optimizer_destroy(&optimizer);
        network_destroy(&nn);
        dataset_destroy(&dataset);

        return 1;
    }

    /* Allocating memory for mnist test data */
    Dataset test_dataset;
    dataset_create(&test_dataset, TEST_IMAGE_COUNT, INPUT_SIZE, OUTPUT_SIZE);

    /* Reading mnist test data */
    emnist_read_image_data_normalized(
        test_dataset.data, test_image, TEST_IMAGE_COUNT);

    emnist_read_label_data(
        test_dataset.label, test_label, TEST_IMAGE_COUNT);

    /* Closing mnist test files */
    fclose(test_image);
    fclose(test_label);

    /*=====================================================================*/

    /* Testing Model Accuracy */
    float accuracy;

    network_evaluate(&accuracy, &nn, &test_dataset);

    printf("\n========== Test Accuracy ==========\n");
    printf("Accuracy:  %.2f%%\n", accuracy);

    /* Clean up */
    optimizer_destroy(&optimizer);
    network_destroy(&nn);
    dataset_destroy(&dataset);
    dataset_destroy(&test_dataset);

    return 0;
}