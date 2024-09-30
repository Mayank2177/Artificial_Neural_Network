Activation functions play a pivotal role in neural networks, introducing non-linearity into the model. This non-linearity is essential for approximating complex patterns and functions, enabling neural networks to learn and represent a wide range of data.
For example:
    for set of input Zi the given weights Wi and bias Bi, the sigmoid function sums all the input to process through the activation function which helps as in optimising the accuracy thus reducing the loss function.
The activation function implies whether the given set of output to be activated on not.
Without non-linearity, a neural network would be equivalent to a single-layer perceptron, capable of only learning linear relationships.
Common Activation Functions
  * Sigmoid: Produces an output between 0 and 1, often used in output layers of binary classification problems.
  * Tanh: Similar to sigmoid but produces outputs between -1 and 1.
  * ReLU (Rectified Linear Unit): Outputs the input if it's positive, otherwise 0. Widely used due to its computational efficiency and ability to avoid the vanishing gradient problem.
  * Leaky ReLU: A variant of ReLU that introduces a small slope for negative inputs, helping to address the dying ReLU problem.
  * Softmax: Often used in the output layer of multi-class classification problems, producing a probability distribution over the possible classes.

## Key Components of the Model:
Input Layer (fc1): This is the first fully connected layer, connecting the input features to the hidden layer. The size of this layer is determined by the input data size.

Hidden Layer (fc2): Another fully connected layer to model the non-linear relationships between features.

Output Layer (fc3): This is the final fully connected layer that outputs a probability distribution over the classes (in this case, 2 classes).

Custom Activation: You've created a CustomActivation class that applies a learnable linear transformation on the layer's output.

Softmax: This is used to convert the network's output into probabilities, but since CrossEntropyLoss already incorporates 
softmax, it can be removed for binary classification tasks.

## Components:
* Input to Hidden Layer (fc1): Maps the input features to a hidden layer.
* Hidden to Hidden Layer (fc2): Models deeper representations.
* Hidden to Output Layer (fc3): Outputs the raw scores (logits) for each class.
* Custom Activation: Applies the learnable linear transformation k1 * x + k2 on the outputs of each layer.
* Softmax: Converts the logits into probabilities for each class.

