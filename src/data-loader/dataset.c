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

#include "data-loader/dataset.h"

#include <stdlib.h>

void dataset_create(Dataset *dataset, usize size, usize input_count, usize class_count) {
    dataset->size        = size;
    dataset->input_count = input_count;
    dataset->class_count = class_count;
    
    dataset->data   = malloc(size * input_count * sizeof(float));
    dataset->label = malloc(size * class_count * sizeof(u8));
    if (!dataset->data || !dataset->label) {
        free(dataset->data);
        free(dataset->label);
        
        dataset->data = NULL;
        dataset->label = NULL;
    }
}

void dataset_destroy(Dataset* dataset) {
    free(dataset->data);
    free(dataset->label);

    dataset->data = NULL;
    dataset->label = NULL;
    dataset->size = 0;
    dataset->input_count = 0;
    dataset->class_count = 0;
}
