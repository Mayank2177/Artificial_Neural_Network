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
