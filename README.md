# Convolutional Autoencoder for Anomaly Detection
This repository is an Tensorflow re-implementation of "Reverse Reconstruction of Anomaly Input Using Autoencoders" from Akihiro Suzuki and Hakaru Tamukoh. The main distinction from the paper is the model included the convolutional related layers to perform better to CIFAR10 dataset. This repository also included the proposed method by Al Aama Obada and Hakaru Tamukoh by converting the images to either HUV or TUV color space instead of RGB space.

## Requirements:
```
tensorflow == 1.12.0
matplotlib >= 3.0.2
numpy >= 1.15.4 
```

## To run the model:
To train and test the model with CIFAR10 dataset: `python main.py`.

## Additional information:
Please refer to the `report.pdf`.

## License:
MIT license

## Citing
```
@InProceedings{aki2018reverse,
  title={Reverse Reconstruction of Anomaly Input Using Autoencoders},
  booktitle={2018 International Symposium on Intelligent Signal Processing and Communication Systems (ISPACS2018), FM1B-3},
  author={Akihiro Suzuki and Hakaru Tamukoh},
  address = {Okinawa, Japan},
  year={2018}
}

@article{tamukoh2020training,
  title={Training Autoencoder using Three Different Reversed Color Models for Anomaly Detection},
  author={Tamukoh, Hakaru and others},
  journal={Journal of Robotics, Networking and Artificial Life},
  year={2020},
  publisher={Atlantis Press}
}
```
