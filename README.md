# Fashion-MNIST Classification

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A deep learning project that classifies fashion items from the Fashion-MNIST dataset using transfer learning with a pre-trained EfficientNetB0 model. THe model can distinguish between 10 different clothing categories including t-shirts, pants, shoes, and accessories with 94% accuracy on the test set.

<div align="center">
    <img src="https://datasets.activeloop.ai/wp-content/uploads/2022/09/Fashion-MNIST-dataset-Activeloop-Platform-visualization-image.webp" alt="Alt text" width="500">
</div>

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/dtiourine/fmnist-classification.git
cd fmnist-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training the Model

The easiest way to train the model from scratch and see its performance on the train/val/test sets is using the Makefile commands:

```bash
# Train the model with default parameters
make train

# Train with custom parameters
make train BATCH_SIZE=128 STAGE1_EPOCHS=5 STAGE2_EPOCHS=12
```

## Project Organization

```
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README 
├── data               <- Data downloaded from source.
│
├── models             <- Trained and serialized models
│
├── notebooks          <- Jupyter notebooks for experimentation and data exploration
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         src and configuration for tools like black
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment
│
├── setup.cfg          <- Configuration file for flake8
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── modeling                
    │   ├── __init__.py
    │   ├── model.py            <- Code to get architecture of the model          
    │   └── train.py            <- Code to train model from scratch and visualize performance on train/val/test sets
    │
    └── plots.py                <- Code to create visualizations
```

## Results

With default settings, you should expect:
- **Training Accuracy**: ~97%
- **Validation Accuracy**: ~94%
- **Test Accuracy**: ~94%

--------

