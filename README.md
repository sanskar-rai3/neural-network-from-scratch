# Neural Network From Scratch in C

[![License: GPL v3 or later](https://img.shields.io/badge/License-GPL%20v3%20or%20later-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.html)

A dense neural network implemented entirely in C, without machine-learning frameworks. The goal is to understand how neural networks work internally by implementing the mathematics and components directly rather than hiding them behind high-level libraries.

Supports matrix operations, EMNIST loading, dense layers, activation functions (ReLU, Leaky ReLU, Sigmoid, Tanh, Softmax), loss functions (MSE, Cross-Entropy), backpropagation, configurable multi-layer networks, SGD optimization, and EMNIST training/inference/evaluation.

## Documentation

Full API and design documentation can be found inside [docs/](./docs/)

## Building

```bash
cmake -S . -B build
cmake --build build
```

Run from the project root so relative EMNIST paths resolve:

```bash
./nn
```

## EMNIST Dataset

The download script supports the following datasets:

* `balanced`
* `byclass`
* `bymerge`
* `digits`
* `letters`
* `mnist`

For example, to download the **Balanced** dataset:

```bash
./script/download_emnist.sh balanced
```

Expected files:

```
data/balanced/
├── emnist-balanced-train-images-idx3-ubyte
├── emnist-balanced-train-labels-idx1-ubyte
├── emnist-balanced-test-images-idx3-ubyte
└── emnist-balanced-test-labels-idx1-ubyte
```

To download **ByClass**:

```bash
./script/download_emnist.sh byclass
```

The EMNIST archive is downloaded only once and reused when downloading additional datasets.

## License
 
This project is licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later).
 
You may use, modify, and redistribute this software under the terms of the GPL-3.0-or-later license.
 
See the LICENSE file for the full license text.
