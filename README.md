## "Fashion MNIST Image Classification with Intel Optimization: A Deep Learning Approach"

## Framework and Tools for Fashion MNIST CNN with Intel Optimization

### Abstract


This report presents a deep learning framework for the Fashion MNIST dataset using a Convolutional Neural Network (CNN). The primary objective is to train a CNN model to classify images of clothing items into ten distinct categories. To optimize the training process, the code leverages Intel's optimization techniques, which are expected to improve the model's convergence speed and performance. The code implements Intel's optimization techniques for deep learning, particularly the Sharpness-Aware Minimization (SAM) optimizer. SAM combines the benefits of SGD (Stochastic Gradient Descent) with an adaptive update strategy, resulting in faster convergence and potentially better performance.
The implementation of image classification task is held using the Fashion MNIST dataset, a popular benchmark in computer vision. The objective is to build a deep learning model capable of classifying fashion items into ten distinct categories. The implementation is conducted in Jupyter Notebook, utilizing Google DevCloud and the Python kernel of the oneAPI. We investigate the impact of Intel Optimization on the model's performance, with a focus on accuracy and computational efficiency. Our findings indicate that the Intel Optimization, applied to the PyTorch framework, leads to significant improvements in the classification task, offering enhanced speed and precision. The study demonstrates the potential benefits of leveraging Intel's optimized tools and libraries for deep learning tasks on Fashion MNIST, and suggests that further exploration of these optimizations could yield valuable insights for more extensive and complex datasets and models.

### Introduction

Fashion MNIST is a widely used dataset in the field of computer vision and machine learning. It comprises 60,000 training images and 10,000 test images of 28x28 grayscale clothing items, each belonging to one of ten classes (e.g., T-shirt, trouser, dress, etc.). The goal of this project is to design a CNN model to accurately classify these images.

### Dependencies

To run the code successfully, the following dependencies need to be installed:

1. PyTorch: PyTorch is a popular deep learning framework that provides a high-level API for building and training neural networks. It supports both CPU and GPU computation and offers various optimization techniques to enhance model performance.

2. NumPy: NumPy is a fundamental package for scientific computing with Python. It provides support for large, multi-dimensional arrays and mathematical functions to operate on these arrays efficiently.

3. torchvision: torchvision is a PyTorch library that provides access to popular datasets, model architectures, and image transformations for computer vision tasks.

4. scikit-learn: scikit-learn is a versatile machine learning library that offers a wide range of tools for data preprocessing, model evaluation, and performance metrics.

5. matplotlib: matplotlib is a plotting library in Python that is widely used for data visualization.
   
6.intel-extension-for-pytorch PyTorch with Intel's Deep Learning Optimization Toolkit

### Framework Structure

The framework is divided into several main sections:

#### Data Preparation

The dataset is first downloaded and transformed into tensors using the torchvision library. Data normalization is performed to ensure that pixel values lie in the range [-1, 1], which helps stabilize training and improve convergence.

#### Model Architecture

The CNN model is designed using PyTorch's nn.Module class. It consists of convolutional layers, max-pooling layers, fully connected layers, and activation functions such as ReLU. The model aims to learn hierarchical features from the input images to make accurate predictions.

#### Intel Optimization

The code implements Intel's optimization techniques for deep learning, particularly the Sharpness-Aware Minimization (SAM) optimizer. SAM combines the benefits of SGD (Stochastic Gradient Descent) with an adaptive update strategy, resulting in faster convergence and potentially better performance.

#### Model Training

The model is trained on the training dataset using the SAM optimizer. The training process is carried out over a specified number of epochs. During training, the model's accuracy and loss are monitored to assess its performance.

#### Hyperparameter Tuning

The code explores different hyperparameter configurations, such as batch size, learning rate, and rho (a hyperparameter specific to SAM), to optimize the model's performance further. The best hyperparameters are identified based on the model's accuracy on the validation set.

#### Model Evaluation

The trained model's performance is evaluated on the test dataset using accuracy and other classification metrics such as precision, recall, and F1-score. ROC curves are also plotted for individual classes, and Intersection over Union (IoU) is calculated to assess multi-class performance.

#### Saving and Loading Models

The optimized model is saved to a file in PyTorch's native format (`.pth`), while the model's weights are saved to an HDF5 file (`.h5`) for compatibility with other platforms.

### Conclusion

In conclusion, this report presents a comprehensive framework for training and evaluating a CNN model on the Fashion MNIST dataset. By leveraging Intel's optimization techniques, the code aims to enhance the model's performance and training speed. The hyperparameter tuning process further refines the model's configuration for better accuracy.

The success of the model is evaluated through various performance metrics, including accuracy, precision, recall, F1-score, ROC curves, and IoU. This enables a thorough assessment of the model's capabilities in classifying clothing images accurately.

Overall, the framework serves as a valuable tool for researchers and practitioners interested in deep learning and computer vision tasks, offering insights into the impact of optimization techniques on model performance and training efficiency. As new optimization algorithms and hardware capabilities emerge, this framework can be easily adapted to explore and experiment with the latest advancements in the field.
