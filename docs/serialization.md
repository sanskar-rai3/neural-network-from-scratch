# Model Serialization Format

The neural network model is serialized as a binary file.

All integer fields are stored as **4-byte unsigned integers**. Matrix elements are stored as **4-byte floating-point values**.

## File Layout

```text
Header
├── Magic number       4 bytes
├── Epochs trained     4 bytes
└── Layer count        4 bytes

Layer 0
├── Weight
│   ├── Rows           4 bytes
│   ├── Columns        4 bytes
│   └── Data            rows × columns × 4 bytes
│
└── Bias
    ├── Rows           4 bytes
    ├── Columns        4 bytes
    └── Data            rows × columns × 4 bytes

Layer 1
├── Weight
│   ├── Rows           4 bytes
│   ├── Columns        4 bytes
│   └── Data            rows × columns × 4 bytes
│
└── Bias
    ├── Rows           4 bytes
    ├── Columns        4 bytes
    └── Data            rows × columns × 4 bytes

...
```

## Header

The file begins with a fixed-size 16-byte header.

| Field          |    Size | Description                                     |
| -------------- | ------: | ----------------------------------------------- |
| Magic number   | 4 bytes | Identifies the file as a neural network model   |
| Epochs trained | 4 bytes | Number of epochs the model has been trained for |
| Layer count    | 4 bytes | Number of layers stored in the file             |

### Header Layout

```text
Offset  Size  Field
0       4     Magic number
8       4     Epochs trained
12      4     Layer count
```

## Layer Format

Each layer is serialized sequentially.

The number of layers is specified by the `layer count` field in the header.

Each layer contains a weight matrix followed by a bias matrix.

### Weight Matrix

```text
Rows       4 bytes
Columns    4 bytes
Data       rows × columns × 4 bytes
```

The matrix data contains `rows × columns` elements.

Each element occupies 4 bytes.

### Bias Matrix

```text
Rows       4 bytes
Columns    4 bytes
Data       rows × columns × 4 bytes
```

The matrix data contains `rows × columns` elements.

Each element occupies 4 bytes.

## Matrix Data Ordering

Matrix elements are stored in **row-major order**.

For a matrix with `rows = 2` and `cols = 3`:

```text
[ a b c ]
[ d e f ]
```

the serialized data is:

```text
a b c d e f
```

The element at `(row, col)` is stored at index:

```text
index = row * cols + col
```

## Example

A model containing two layers could be represented as:

```text
Header
  magic number
  epochs trained
  layer count = 2

Layer 0
  weight.rows
  weight.cols
  weight.data
  bias.rows
  bias.cols
  bias.data

Layer 1
  weight.rows
  weight.cols
  weight.data
  bias.rows
  bias.cols
  bias.data
```

The layers are stored in the same order as they appear in the network.

## Binary Representation

All fields are written consecutively with no padding between fields.

For a matrix:

```text
rows
cols
element[0]
element[1]
...
element[rows * cols - 1]
```

The total serialized size of a matrix is:

```text
8 + (rows × cols × 4) bytes
```

The total serialized size of one layer is therefore:

```text
16
+ (weight.rows × weight.cols × 4)
+ (bias.rows × bias.cols × 4)
bytes
```

The total model size is:

```text
16
+ Σ [
    16
    + (weight.rows × weight.cols × 4)
    + (bias.rows × bias.cols × 4)
  ]
bytes
```

where the sum contains one term for every layer.

## Compatibility

The `magic number` is used to verify that the file is a model file.
