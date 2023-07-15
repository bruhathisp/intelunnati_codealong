## CNN Architecture 

1. Convolutional Layer 1:
   - Input: Single-channel grayscale images with a size of 28x28 pixels.
   - Output: 16 feature maps, obtained by applying 16 filters of size 3x3 to the input image.
   - Activation Function: ReLU (Rectified Linear Unit) is applied to the output of the convolutional layer to introduce non-linearity.

  2. Max-Pooling Layer 1:
   - Max-pooling is applied to reduce the spatial dimensions of the feature maps.
   - The pooling operation is performed with a kernel size of 2x2 and a stride of 2, resulting in a halving of the spatial dimensions.

  3. Convolutional Layer 2:
   - Input: 16 feature maps from the previous layer.
   - Output: 32 feature maps, obtained by applying 32 filters of size 3x3 to the input feature maps.
   - Activation Function: ReLU.

4. Max-Pooling Layer 2:
   - Max-pooling with a kernel size of 2x2 and a stride of 2, reducing the spatial dimensions further.

5. Flattening Layer:
   - The output of the last max-pooling layer is flattened into a 1-dimensional vector, preparing it for the fully connected layers.

6. Fully Connected Layer 1 (Dense Layer):
   - Input: The flattened vector representing the 32x7x7 feature maps (32 channels with 7x7 spatial dimensions).
   - Output: 128 neurons in the hidden layer.
   - Activation Function: ReLU.

7. Fully Connected Layer 2 (Output Layer):
   - Input: 128 neurons from the previous hidden layer.
   - Output: 10 neurons representing the class probabilities for each of the 10 Fashion MNIST categories (e.g., T-shirt/top, Trouser, Pullover, etc.).
   - No activation function is applied to the output layer since it represents raw class scores.

The CNN architecture follows a common pattern used for image classification tasks. 
Convolutional layers help to extract relevant features from the input images, and the max-pooling layers reduce spatial dimensions to retain the most
important information. The fully connected layers at the end of the network perform the final classification based on the learned features.

Overall, this architecture can provide reasonable performance for the Fashion MNIST dataset, and the model can be further optimized using various
techniques like learning rate scheduling, data augmentation, and early stopping during training. The SAM optimizer and Intel optimization further enhance the 
model's training process, leading to potential improvements in accuracy and convergence.


## Approach:

### 1. Convolutional Neural Network (CNN) Architecture:
The code provided defines a simple Convolutional Neural Network (CNN) architecture for classifying Fashion MNIST images. 
The model consists of two convolutional layers with ReLU activation functions, followed by max-pooling layers to reduce spatial dimensions. 
The output of the last max-pooling layer is then flattened and passed through two fully connected (dense) layers with ReLU activations, ending with a final 
output layer of 10 neurons representing the class probabilities for each Fashion MNIST category.

The architecture of the CNN is as follows:

```python
class FashionMNISTCNN(nn.Module):
    def __init__(self):
        super(FashionMNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

### 2. Data Preparation:
Fashion MNIST dataset is used for training, validation, and testing. The training dataset is further divided into train and validation sets.
The images are preprocessed by transforming them into tensors and then normalizing the pixel values.

### 3. Optimizer: Sharpness-Aware Minimization (SAM)
The code defines a custom optimizer called `SAM` (Sharpness-Aware Minimization). SAM is an optimization technique that aims to minimize a modified 
loss function, taking into account the sharpness of the loss landscape. SAM modifies the gradients of the parameters before the optimizer step by adding 
a small multiple of the current parameter values to the gradients. This helps in improving optimization convergence and generalization.

### 4. Intel Optimization:
The code leverages Intel Extension for PyTorch (IPEX) for optimizing the model and the optimizer. IPEX is designed to accelerate PyTorch workloads on 
Intel processors. The `ipex.optimize` function is used to optimize the model and optimizer with the specified dtype (`torch.float32`).

### 5. Model Training and Hyperparameter Tuning:
The code performs training with the SAM optimizer and Intel optimization. It runs the training loop for the specified number of epochs and records training 
and validation losses, accuracies, and hyperparameters for each run. It then selects the best set of hyperparameters based on validation accuracy for future 
evaluations.

### 6. Model Evaluation:
After training, the code evaluates the optimized model on the test dataset and calculates the test accuracy. Additionally, it generates ROC curves and 
computes the Intersection over Union (IoU) for multi-class classification. The IoU provides a measure of how well the model is performing on each class. 
Classification report is generated to get the prediction accuracy score report.

### 7. Model Summary:
Finally, the code provides a summary of the model architecture using `torchsummary.summary` package.

## Math Behind Sharpness-Aware Minimization (SAM):

The Sharpness-Aware Minimization (SAM) optimizer is based on the theory that optimization can be improved by taking into account the sharpness of the loss 
landscape. Traditional optimization methods like SGD and Adam aim to find the steepest direction to minimize the loss. However, in regions with a sharp loss
landscape, the gradients can be dominated by noise and lead to poor optimization.

SAM modifies the gradients by adding a small multiple (controlled by the hyperparameter `rho`) of the current parameter values to the gradients.
This operation makes the gradients point in a direction that takes into account the current sharpness of the loss landscape. By doing so, it aims to
improve the convergence and generalization of the optimization process.

The SAM optimizer updates can be described by the following equations:

1. Calculate the gradients:
```
grad = dL/dw   # Gradients of the loss w.r.t. model parameters
```

2. Calculate the modified gradients:
```
modified_grad = grad + rho * w    # Add a small multiple (rho) of the current parameter values to the gradients
```

3. Perform the optimizer step using the modified gradients:
```
w_new = w - lr * modified_grad   # Update the model parameters using the modified gradients and the learning rate
```

Where:
- `grad`: Gradients of the loss w.r.t. model parameters.
- `w`: Current model parameters.
- `rho`: A hyperparameter controlling the strength of the SAM modification.
- `lr`: Learning rate.

The SAM optimizer is used in conjunction with a base optimizer (e.g., SGD, Adam) that performs the actual parameter updates. 
The base optimizer is used to update the model parameters after the gradients are modified using SAM.

Note: The SAM optimizer in the provided code falls back to non-fused step because fused step is not supported for SAM by Intel Extension for PyTorch (IPEX).

## Conclusion:
The provided code demonstrates how to define, train, optimize, and evaluate a simple Convolutional Neural Network (CNN) for the Fashion MNIST dataset 
using the Sharpness-Aware Minimization (SAM) optimizer and Intel Optimization. It also showcases how to generate ROC curves,
compute Intersection over Union (IoU) for multi-class classification, and generate a classification report for prediction accuracy. 
Additionally, the code provides a summary of the model architecture.
