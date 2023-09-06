## Project README

This project aims to create an algorithm that converts a grayscale image to a color image using the CIFAR10 dataset. The grayscale image is defined as the average of the three color channels.

### Dependencies

To run this project, you need the following dependencies:

- `torchvision`
- `torch`
- `matplotlib`
- `tqdm`

You can install the dependencies using the following command:

```
pip install torchvision torch matplotlib tqdm
```

### Dataset

The CIFAR10 dataset is used in this project. It consists of 50,000 training images and 10,000 test images, with a total of 10 different classes. The dataset is automatically downloaded and saved in the `./datasets` directory.

### Preprocessing

The grayscale transformation is performed using the `transforms.Grayscale` class from the `torchvision.transforms` module. The grayscale image is then resized to (32, 32) using `transforms.Resize` and converted to a tensor using `transforms.ToTensor`.

### Model Architecture

Two different colorization models are implemented in this project.

#### Model 1: CNVC

The `CNVC` model is defined as follows:

```
class CNVC(nn.Module):
  def __init__(self):
      super(CNVC, self).__init__()
      # Convolutional layers for colorization

  def forward(self, x):
      # Forward pass implementation
```

The model consists of a series of convolutional layers for colorization. The grayscale image is passed through these layers to generate the colorized image.

#### Model 2: CNVC2

The `CNVC2` model is defined as follows:

```
class CNVC2(nn.Module):
  def __init__(self):
      super(CNVC2, self).__init__()
      # Convolutional and transposed convolutional layers for colorization

  def forward(self, x):
      # Forward pass implementation
```

The model consists of both convolutional and transposed convolutional layers for colorization. The grayscale image is passed through these layers to generate the colorized image.

### Training

The models are trained using the Mean Squared Error (MSE) loss and the Adam optimizer. The training loop is run for a specified number of epochs, and the loss is calculated for each batch. The optimizer is updated based on the loss gradients to minimize the loss.

### Results

After training the models, the `compareResults` function is used to display the results. It takes a model as input and generates colorized images from the test dataset. The colorized images, grayscale images, and original images are displayed side by side for comparison.

### Usage

To use this project, follow these steps:

1. Install the required dependencies mentioned above.
2. Run the code in a Python environment that supports Jupyter Notebook or JupyterLab.
3. Make sure the CIFAR10 dataset is downloaded and saved in the `./datasets` directory.
4. Run the code cells in the Jupyter Notebook or JupyterLab environment to train the models and display the results.

Note: The code assumes that a CUDA-enabled GPU is available for faster training. If a GPU is not available, the code will run on the CPU, but the training process may be slower.

### Conclusion

This project demonstrates the process of converting grayscale images to color images using two different colorization models. By training these models on the CIFAR10 dataset, it is possible to generate colorized images from grayscale input. The results can be further improved by experimenting with different model architectures, hyperparameters, and training strategies.