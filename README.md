# Convolutional Anomaly Detection Autoencoder
This is an implementation of "Reverse Reconstruction of Anomaly Input Using Autoencoders" from Akihiro Suzuki et al. The main distinction between this implementation and the paper is I changed the model from all fully connected layers to convolution related layers to increasing the capacity of model. This work also included the TUV space converting proposed by Al Aama Obada (Currently, the code did not supported).

## Requirements:
  1. Python 3.6.8
  2. Tensorflow 1.12.0
  3. Matplotlib 3.0.2
  4. Numpy 1.15.4 

## Run
To train the model and test after training: `python main.py`.

## Citing
```
@InProceedings{aki2018reverse,
  title={Reverse Reconstruction of Anomaly Input Using Autoencoders},
  booktitle={2018 International Symposium on Intelligent Signal Processing and Communication Systems (ISPACS2018), FM1B-3},
  author={Akihiro Suzuki and Hakaru Tamukoh},
  address = {Okinawa, Japan},
  year={2018}
}
```
