This is the code for our publication "Understanding and Simplifying Architecture Search in Spatio-Temporal Graph Neural Networks" accepted by Transactions on Machine Learning Research (**TMLR**) 2023.

# Dataset

- PeMS03/04/07/08

These four datasets are available from [Spatial-Temporal Synchronous Graph Convolutional Networks: A New Framework for Spatial-Temporal Network Data Forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/5438)

- NE-BJ

This dataset is available from [Dynamic Graph Convolutional Recurrent Network for Traffic Prediction: Benchmark and Solution](https://arxiv.org/abs/2104.14917)

# Environment

We use minimal environment requirement as below. Our code is tested on `CUDA 11.1`

```
numpy
pandas
torch==1.9.0
```

# Run

```
cd SimpleSTG
python run.py
```
More configurations other than default can be found in `run.py`.
