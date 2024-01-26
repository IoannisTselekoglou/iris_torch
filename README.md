# Iris Classifcation

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#Dataset)
- [Model](model)
- [Installation](#installation)
- [Examples](#examples)
- [License](#license)

## Introduction

Welcome to Iriis Torch, a project using PyTorch for iris flower classification. This project includes a dataset, a model for iris flower classification and a Jupyter notebook for analysis and visualization.

## Data set

The dataset used in this notebook is available on Kaggle.com [here](https://www.kaggle.com/datasets/uciml/iris)

The dataset for this project contains information about iris flowers. It includes features such as sepal length, sepal width, petal length, and petal width, along with the corresponding labels indicating the species of the iris flower.


## Model 

The implemented model is a very simple neural network. The model is implemented using the `NeuralNetwork` class.

### Model Layers
1. **Input Layer:** 4 nodes
   - Input features: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm

2. **Hidden Layer 1:** 
   - Architecture: Linear layer with 128 nodes and ReLU activation function

3. **Hidden Layer 2:** 
   - Architecture: Linear layer with 64 nodes and ReLU activation function

4. **Output Layer:** 3 nodes
   - Output categories: Iris species


## Example 

The iris_notebook.ipynb Jupyter notebook is a walks you through the entire process of loading data, training the model, and evaluating its performance. It also includes visualizations and analysis of the results to provide insights into the iris flower classification.

## Installation

1. Clone this repository.
2. Download Dataset.
3. Install the required dependencies using pip:
    ```bash
    - pip install matplotlib
    - pip install numpy 
4. Run Jupyter Notebook. 

## License

- [NumPy](https://numpy.org): A library for numerical computing in Python. License: [BSD 3-Clause License](licenses/LICENSE-numpy.txt).
- [Matplotlib](https://matplotlib.org): A plotting library for Python. License: [Matplotlib License](licenses/LICENSE-matplotlib.txt).



