TODO:

- experiment with batching
- test gru in place of LSTM
- learn to implement multi-branch models with the Keras functional API
- convert data reading to use Dataset objects
- experimemnt with different number of convolution layers in front of LSTMs
- add attention to lstm (attention package one straightforward option)
- feature engineering
  -hand-crafted statistics (mean, std, min, max, percentiless, ample covariance matrix)

  - the same normalized to dataset mean
  - component-wise power spectra
  - add lag1 differences as channels

- set up MLP based on engineered features
- set up LGBM
- set up Cuda/GPU on the Windows desktop
  - conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
