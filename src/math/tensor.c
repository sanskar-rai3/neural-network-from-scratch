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
#include "math/tensor.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>


// error handling will be implemented later

void tensor_create(Tensor *t, usize rank, const usize *shape) {
    t->rank = rank;

    t->shape = malloc(rank * sizeof(usize));
    if (!t->shape) {
        // error handling
    }

    memcpy(t->shape, shape, rank * sizeof(usize));

    t->size = 1;
    for (usize i = 0; i < rank; i++) {
        t->size *= t->shape[i];
    }

    t->strides = malloc(rank * sizeof(usize));
    if (!t->strides) {
        // error handling
    }

    t->strides[rank - 1] = 1;
    for (usize i = rank - 1; i > 0; i--) {
        t->strides[i - 1] = t->strides[i] * t->shape[i];
    }

    t->data = malloc(t->size * sizeof(float));
    if (!t->data) {
        // error handling
    }
}

void tensor_destroy(Tensor *t) {
    free(t->data);
    free(t->shape);
    free(t->strides);

    t->size    = 0;
    t->rank    = 0;
    t->shape   = NULL;
    t->strides = NULL;
}

void tensor_copy(Tensor *dest, const Tensor *src) {
    tensor_create(dest, src->rank, src->shape);    
    memcpy(dest->data, src->data, src->size * sizeof(usize));
}

void tensor_fill(Tensor *t, float val) {
    for (usize i = 0; i < t->size; i++) {
        t->data[i] = val;
    }
}

void tensor_get(float *val, const Tensor *t, const usize *index) {
    int offset = 0;
    for (usize i = 0; i < t->rank; i++) {
        index += t->strides[i] * index[i]; 
    }

    *val = t->data[offset];
}

void tensor_set(Tensor *t, const usize *index, float val) {
    int offset = 0;
    for (usize i = 0; i < t->rank; i++) {
        index += t->strides[i] * index[i]; 
    }

    t->data[offset] = val;
} 

void tensor_max(float *ret, const Tensor *t) {
    float max = t->data[0];

    for (usize i = 1; i < t->size; i++) {
        if (t->data[i] > max) {
            max = t->data[i];
        }
    }

    *ret = max;
}

void tensor_min(float *ret, const Tensor *t) {
    float min = t->data[0];

    for (usize i = 1; i < t->size; i++) {
        if (t->data[i] < min) {
            min = t->data[i];
        }
    }

    *ret = min;
}

void tensor_sum(float *ret, const Tensor *t) {
    float sum = 0.0f;

    for (usize i = 0; i < t->size; i++) {
        sum += t->data[i];
    }

    *ret = sum;
}

void tensor_hadamard_multiply(Tensor *result, const Tensor *a, const Tensor *b) {
    for (usize i = 0; i < result->size; i++) {
        result->data[i] = a->data[i] * b->data[i];
    }
}

void tensor_scalar_multiply(Tensor *result, const Tensor *t, float scalar) {
    for (usize i = 0; i < result->size; i++) {
        result->data[i] = scalar * t->data[i];
    }
}

static inline float random_float(float min, float max) {
    static bool seeded = false;
    if (!seeded) {
        srand(time(NULL));
        seeded = true;
    }

    float scale = (float)rand() / (float)RAND_MAX;
    return min + scale * (max - min);
}

void tensor_randomize(Tensor *t, float min, float max) {
    for (usize i = 0; i < t->size; i++) {
        t->data[i] = random_float(min, max);
    }
}

void tensor_he_uniform(Tensor *t, usize fan_in) {
    float limit = sqrtf(6.0f / (float)fan_in);
    tensor_randomize(t, -limit, limit);
}

void tensor_xavier_uniform(Tensor *t, usize fan_in, usize fan_out) {
    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    tensor_randomize(t, -limit, limit);
}

void tensor_flatten(Tensor *t) {
    t->rank = 1;

    free(t->strides);
    t->strides = malloc(sizeof(usize));
    if (!t->strides) {
        // error handling
    }

    t->strides[0] = 1;

    free(t->shape);
    t->shape = malloc(sizeof(usize));
    if (!t->shape) {
        // error handling
    }

    t->shape[0] = t->size;
}

void tensor_reshape(Tensor *t, usize rank, const usize *shape) {
    usize size = 1;

    for (usize i = 0; i < rank; i++)
        size *= shape[i];

    if (size != t->size) {
        return;
    }

    t->rank = rank;

    t->shape = malloc(rank * sizeof(usize));
    if (!t->shape) {
        // error handling
    }

    memcpy(t->shape, shape, rank * sizeof(usize));

    t->size = 1;
    for (usize i = 0; i < rank; i++) {
        t->size *= t->shape[i];
    }

    t->strides = malloc(rank * sizeof(usize));
    if (!t->strides) {
        // error handling
    }

    t->strides[rank - 1] = 1;
    for (usize i = rank - 1; i > 0; i--) {
        t->strides[i - 1] = t->strides[i] * t->shape[i];
    }
}
