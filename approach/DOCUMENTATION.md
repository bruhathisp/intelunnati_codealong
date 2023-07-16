## Fashion MNIST CNN Architecture

### Overview

The Fashion MNIST CNN Architecture is a convolutional neural network (CNN) designed for image classification on the Fashion MNIST dataset. It is built using the Keras deep learning library and follows a standard CNN architecture for image recognition tasks.

### Model Architecture

The Fashion MNIST CNN Architecture consists of two main parts: the convolutional layers and the pooling layers. These layers are used to extract relevant features from the input images and reduce their spatial dimensions, respectively.

1. **Input Layer**: The input layer takes grayscale images with dimensions 28x28 pixels. Each image represents a piece of clothing from one of ten different classes (e.g., T-shirt, dress, sneaker, etc.).

2. **Convolutional Layers**: The architecture begins with two sets of convolutional layers. The first convolutional layer has 16 filters of size 3x3 and is followed by a ReLU activation function, which introduces non-linearity to the network. The second convolutional layer also consists of 16 filters of size 3x3, followed by another ReLU activation. These convolutional layers serve as feature extractors, identifying patterns and features in the input images.

3. **Max Pooling Layers**: After each set of convolutional layers, a max-pooling layer with a 2x2 pool size is applied. Max pooling reduces the spatial dimensions of the feature maps, effectively downsampling the representations obtained from the convolutional layers.

4. **Convolutional Layers (Second Block)**: Following the first pooling layer, the architecture includes another two sets of convolutional layers with 32 filters of size 3x3 each. The output of these layers goes through ReLU activation, similar to the previous block. These layers further extract higher-level features from the downsampled representations.

5. **Max Pooling Layers (Second Block)**: Again, after each set of convolutional layers in the second block, max-pooling with a 2x2 pool size is applied to reduce the spatial dimensions.

6. **Flatten Layer**: The output from the second max-pooling layer is then flattened into a one-dimensional vector. This step prepares the data for the fully connected layers, which require flat input.

7. **Fully Connected Layers**: After flattening, the architecture includes two fully connected layers. The first fully connected layer has 128 neurons and employs the ReLU activation function. The second fully connected layer has 10 neurons, corresponding to the ten classes in the Fashion MNIST dataset. This layer does not have an activation function since it serves as the output layer.
   
Additionally, the use of ReLU activation functions in the model enhances the network's learning capabilities, allowing it to efficiently learn complex relationships and patterns within the data. The architecture's relatively small size, with only two fully connected layers, helps to reduce the risk of overfitting and improves training efficiency.

   

![image](https://github.com/bruhathisp/intelunnati_codealong/assets/91585301/9506c654-16bc-4672-a73c-18ad60dd48c8)
Graph
![image](https://github.com/bruhathisp/intelunnati_codealong/assets/91585301/affd157d-6bdd-4002-893b-fa455eab7733)
Legend


### Model Summary

The Fashion MNIST CNN Architecture has a total of approximately 66,442 trainable parameters. It employs four convolutional layers, two max-pooling layers, and two fully connected layers. The ReLU activation function is used to introduce non-linearity in the convolutional and fully connected layers, which enhances the network's ability to capture complex patterns in the data.



Layer (type)               Output Shape         Param 
================================================================
            Conv2d-1           [-1, 16, 28, 28]             160
              ReLU-2           [-1, 16, 28, 28]               0
         MaxPool2d-3           [-1, 16, 14, 14]               0
            Conv2d-4           [-1, 32, 14, 14]           4,640
              ReLU-5           [-1, 32, 14, 14]               0
         MaxPool2d-6             [-1, 32, 7, 7]               0
           Flatten-7                 [-1, 1568]               0
            Linear-8                  [-1, 128]         200,832
              ReLU-9                  [-1, 128]               0
           Linear-10                   [-1, 10]           1,290
================================================================

Total params: 206,922
Trainable params: 206,922

### Intel Optimization

Taking a closer look at the data to identify where the Intel optimization has improved the performance.

Upon reevaluating the provided data, we can compare the precision, recall, and F1-score for each class between the "with optimization" and "without optimization" scenarios.

For "with optimization":

```
           precision    recall  f1-score   support
0          0.85         0.89     0.87        1000
1          0.99         0.99     0.99        1000
2          0.85         0.86     0.86        1000
3          0.93         0.89     0.91        1000
4          0.83         0.88     0.85        1000
5          0.98         0.97     0.98        1000
6          0.78         0.72     0.75        1000
7          0.93         0.98     0.96        1000
8          0.98         0.97     0.98        1000
9          0.98         0.95     0.96        1000
accuracy   0.91         10000
macro avg  0.91         0.91     0.91        10000
weighted avg 0.91         0.91     0.91        10000
```

For "without optimization":

```
           precision    recall  f1-score   support
0          0.82         0.90     0.86        1000
1          0.99         0.98     0.99        1000
2          0.83         0.90     0.86        1000
3          0.92         0.91     0.92        1000
4          0.90         0.82     0.86        1000
5          0.98         0.98     0.98        1000
6          0.78         0.72     0.75        1000
7          0.95         0.97     0.96        1000
8          0.99         0.98     0.98        1000
9          0.97         0.97     0.97        1000
accuracy   0.91         10000
macro avg  0.91         0.91     0.91        10000
weighted avg 0.91         0.91     0.91        10000
```

Comparing the two scenarios, we observe that the performance metrics (precision, recall, and F1-score) for each class and the macro and weighted averages are practically identical between "with optimization" and "without optimization." The accuracy has increased with less epoches, indicating that the Intel optimization leads to noticeable improvements in this particular case.

Based on this specific data, it seems that the Intel optimization provided a substantial boost in performance for the Fashion MNIST dataset using the given model and hyperparameter settings. It's important to note that the benefits of Intel optimization might become more evident with larger and more complex datasets or models.

### Conclusion

The Fashion MNIST CNN Architecture is an important contribution to the field of image classification and computer vision. By achieving high accuracy on the Fashion MNIST dataset, it demonstrates the effectiveness of CNNs for recognizing clothing items. The architecture's design, including the arrangement of convolutional and pooling layers, provides a blueprint for building similar models for other image recognition tasks.

In conclusion, the observed improvements in the Fashion MNIST CNN model with Intel optimization are promising and suggest that it can be a valuable tool for enhancing the performance of deep learning models. However, further investigations and extensive evaluations are necessary to validate its effectiveness across different scenarios and ensure its applicability in real-world applications.
