const quiz = [
    {
        tags: [],
        number: 1,
        question: "What is a neural network primarily inspired by?",
        options: [
            {
                letter: "a",
                answer: "Brain's synaptic connections",
            },
            {
                letter: "b",
                answer: "Electrical circuits",
            },
            {
                letter: "c",
                answer: "Cloud computing systems",
            },
            {
                letter: "d",
                answer: "DNA sequencing",
            },
        ],
        correct_answers: ["A"],
        answers: [
            {
                letter: "a",
                answer: "Brain's synaptic connections",
            },
        ],
        explanation:
            "Neural networks are fundamentally inspired by the biological neural networks in the brain.  The interconnected neurons and their synaptic connections, which transmit signals, form the basis of the artificial neural network architecture.  Options B, C, and D are not the primary inspiration for neural networks.",
    },
    {
        tags: [],
        number: 2,
        question: "What does a neuron in a neural network do?",
        options: [
            {
                letter: "a",
                answer: "Stores data",
            },
            {
                letter: "b",
                answer: "Computes a weighted sum and applies an activation function",
            },
            {
                letter: "c",
                answer: "Transmits raw input without modification",
            },
            {
                letter: "d",
                answer: "Predicts future outcomes",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "Computes a weighted sum and applies an activation function",
            },
        ],
        explanation:
            "A neuron in a neural network receives weighted inputs, sums them, and then applies an activation function to produce an output. This output is then passed to other neurons in the network. Options A, C, and D describe functions that are not the core function of a single neuron.",
    },
    {
        tags: ["activation"],
        number: 3,
        question: "What is the role of the activation function?",
        options: [
            {
                letter: "a",
                answer: "Initialize weights",
            },
            {
                letter: "b",
                answer: "Introduce non-linearity to the model",
            },
            {
                letter: "c",
                answer: "Calculate gradients",
            },
            {
                letter: "d",
                answer: "Improve training speed",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "Introduce non-linearity to the model",
            },
        ],
        explanation:
            "Without activation functions, a neural network would simply be a linear transformation of the input data, severely limiting its capacity to learn complex patterns.  Activation functions introduce non-linearity, enabling the network to approximate any continuous function (Universal Approximation Theorem). Options A, C, and D are related to neural network training but not the core role of the activation function.",
    },
    {
        tags: ["activation"],
        number: 4,
        question: "Which of the following is a popular activation function?",
        options: [
            {
                letter: "a",
                answer: "ReLU",
            },
            {
                letter: "b",
                answer: "Heaviside",
            },
            {
                letter: "c",
                answer: "Entropy",
            },
            {
                letter: "d",
                answer: "Gradient",
            },
        ],
        correct_answers: ["A"],
        answers: [
            {
                letter: "a",
                answer: "ReLU",
            },
        ],
        explanation:
            "ReLU (Rectified Linear Unit) is a very popular activation function due to its computational efficiency and effectiveness in mitigating the vanishing gradient problem.  Heaviside is a step function, less commonly used in modern deep learning. Entropy is a measure of uncertainty, not an activation function. Gradient is a mathematical concept related to optimization, not an activation function.",
    },
    {
        tags: ["propagation"],
        number: 5,
        question: 'The term "feedforward" in a neural network means:',
        options: [
            {
                letter: "a",
                answer: "Signals move in a loop",
            },
            {
                letter: "b",
                answer: "Signals move in one direction, input to output",
            },
            {
                letter: "c",
                answer: "Backward propagation of signals",
            },
            {
                letter: "d",
                answer: "Weights are updated continuously",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "Signals move in one direction, input to output",
            },
        ],
        explanation:
            "In a feedforward neural network, information flows in one direction, from the input layer through the hidden layers to the output layer. There are no loops or cycles in the signal flow.  Backward propagation is a separate process used during training, not the definition of feedforward. Continuous weight updates happen during training but don't define the term 'feedforward'.",
    },
    {
        tags: ["training"],
        number: 6,
        question: "What is backpropagation used for?",
        options: [
            {
                letter: "a",
                answer: "Initializing weights",
            },
            {
                letter: "b",
                answer: "Computing the loss function",
            },
            {
                letter: "c",
                answer: "Calculating gradients for weight updates",
            },
            {
                letter: "d",
                answer: "Normalizing the input data",
            },
        ],
        correct_answers: ["C"],
        answers: [
            {
                letter: "c",
                answer: "Calculating gradients for weight updates",
            },
        ],
        explanation:
            "Backpropagation is an algorithm used to calculate the gradients of the loss function with respect to the weights of the neural network. These gradients are then used to update the weights during the training process, aiming to minimize the loss function and improve the model's accuracy.  Options A, B, and D are incorrect; weight initialization is a separate process, the loss function measures error, and data normalization is a preprocessing step.",
    },
    {
        tags: ["training"],
        number: 7,
        question: "The loss or cost function is used to:",
        options: [
            {
                letter: "a",
                answer: "Visualize the network's architecture",
            },
            {
                letter: "b",
                answer: "Determine the error in predictions",
            },
            {
                letter: "c",
                answer: "Improve activation functions",
            },
            {
                letter: "d",
                answer: "Increase model complexity",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "Determine the error in predictions",
            },
        ],
        explanation:
            "The loss or cost function quantifies the difference between the model's predictions and the actual target values.  Minimizing this function is the primary goal of training a neural network. Options A, C, and D are incorrect; the loss function doesn't visualize architecture, improve activation functions, or directly increase model complexity.",
    },
    {
        tags: ["gradient"],
        number: 8,
        question: "Which of the following describes gradient descent?",
        options: [
            {
                letter: "a",
                answer: "A method for initializing weights",
            },
            {
                letter: "b",
                answer: "A method to minimize the loss function",
            },
            {
                letter: "c",
                answer: "A method for visualizing activations",
            },
            {
                letter: "d",
                answer: "A method for testing model performance",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "A method to minimize the loss function",
            },
        ],
        explanation:
            "Gradient descent is an iterative optimization algorithm used to find the minimum of a function (in this case, the loss function) by iteratively moving in the direction of the negative gradient. Options A, C, and D describe unrelated processes.",
    },
    {
        tags: ["gradient", "training"],
        number: 9,
        question: "In stochastic gradient descent (SGD), the weights are updated:",
        options: [
            {
                letter: "a",
                answer: "After processing the entire dataset",
            },
            {
                letter: "b",
                answer: "Using a subset of the dataset (batch)",
            },
            {
                letter: "c",
                answer: "After every single data point",
            },
            {
                letter: "d",
                answer: "At the end of multiple epochs",
            },
        ],
        correct_answers: ["C"],
        answers: [
            {
                letter: "c",
                answer: "After every single data point",
            },
        ],
        explanation:
            "Stochastic Gradient Descent (SGD) updates the model's weights after processing each individual data point. This contrasts with batch gradient descent (using a batch of data points) and mini-batch gradient descent (using a small batch of data points).  The iterative nature of SGD allows for faster updates and can escape local minima more effectively than batch gradient descent, although it introduces more noise in the gradient estimation.",
    },
    {
        tags: ["gradient"],
        number: 10,
        question: "Why is gradient descent performed iteratively?",
        options: [
            {
                letter: "a",
                answer: "It ensures faster convergence",
            },
            {
                letter: "b",
                answer: "Exact solutions cannot be computed directly",
            },
            {
                letter: "c",
                answer: "It introduces random noise to improve results",
            },
            {
                letter: "d",
                answer: "It adjusts the network structure",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "Exact solutions cannot be computed directly",
            },
        ],
        explanation:
            "Gradient descent is iterative because finding the global minimum of a complex loss function analytically is often intractable. The iterative approach allows for an approximation of the minimum through successive steps. Options A, C, and D are incorrect; while iterative methods can lead to faster convergence in some cases, it's not the primary reason for their use.  Random noise is not inherently introduced, and the network structure is not adjusted during gradient descent itself.",
    },
    {
        tags: ["training"],
        number: 11,
        question: "What do weights in a neural network represent?",
        options: [
            {
                letter: "a",
                answer: "Biases for each layer",
            },
            {
                letter: "b",
                answer: "Connections and their importance between neurons",
            },
            {
                letter: "c",
                answer: "Random noise added for regularization",
            },
            {
                letter: "d",
                answer: "Layers of a network",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "Connections and their importance between neurons",
            },
        ],
        explanation:
            "Weights in a neural network represent the strength or importance of the connections between neurons in different layers.  Each connection between a neuron in one layer and a neuron in the next layer has an associated weight.  These weights are learned during the training process and determine how much influence each neuron has on the neurons in the subsequent layer.  Larger weights indicate a stronger connection and greater influence.",
    },
    {
        tags: ["training"],
        number: 12,
        question: "Bias in a neuron helps the model to:",
        options: [
            {
                letter: "a",
                answer: "Avoid underfitting",
            },
            {
                letter: "b",
                answer: "Shift the activation function",
            },
            {
                letter: "c",
                answer: "Decrease the loss value directly",
            },
            {
                letter: "d",
                answer: "Regularize training",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "Shift the activation function",
            },
        ],
        explanation:
            "Bias in a neuron is an additional parameter added to the weighted sum of inputs before the activation function is applied.  It allows the activation function to be shifted along the x-axis. This shift is crucial because without it, the activation function would always pass through the origin (0,0), limiting the model's ability to learn non-linear relationships.  The bias provides flexibility, enabling the neuron to activate even when the weighted sum of inputs is close to zero.",
    },
    {
        tags: ["activation"],
        number: 13,
        question: "Scenario: Suppose ReLU is applied to the input **-3.2**. What will the output be?",
        options: [
            {
                letter: "a",
                answer: "-3.2",
            },
            {
                letter: "b",
                answer: "3.2",
            },
            {
                letter: "c",
                answer: "0",
            },
            {
                letter: "d",
                answer: "1",
            },
        ],
        correct_answers: ["C"],
        answers: [
            {
                letter: "c",
                answer: "0",
            },
        ],
        explanation:
            "The ReLU (Rectified Linear Unit) activation function is defined as max(0, x).  Therefore, if the input is -3.2, the output will be max(0, -3.2) = 0.  ReLU outputs the input if it's positive and 0 otherwise.",
    },
    {
        tags: ["gradient", "training"],
        number: 14,
        question: "In SGD, what is the role of the learning rate?",
        options: [
            {
                letter: "a",
                answer: "It determines how large the weight updates are",
            },
            {
                letter: "b",
                answer: "It increases the model accuracy directly",
            },
            {
                letter: "c",
                answer: "It regularizes the network's weights",
            },
            {
                letter: "d",
                answer: "It eliminates redundant neurons",
            },
        ],
        correct_answers: ["A"],
        answers: [
            {
                letter: "a",
                answer: "It determines how large the weight updates are",
            },
        ],
        explanation:
            "In Stochastic Gradient Descent (SGD), the learning rate is a hyperparameter that controls the step size taken during the weight update process. A smaller learning rate leads to smaller weight adjustments, resulting in slower convergence but potentially a more precise solution. Conversely, a larger learning rate leads to larger weight adjustments, potentially resulting in faster convergence but also a risk of overshooting the optimal solution and failing to converge.",
    },
    {
        tags: ["training"],
        number: 15,
        question: "What happens if the learning rate is set too high?",
        options: [
            {
                letter: "a",
                answer: "The model converges faster",
            },
            {
                letter: "b",
                answer: "The model fails to converge",
            },
            {
                letter: "c",
                answer: "The gradients vanish completely",
            },
            {
                letter: "d",
                answer: "Training becomes more accurate",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "The model fails to converge",
            },
        ],
        explanation:
            "If the learning rate is set too high, the weight updates will be excessively large. This can cause the optimization algorithm to overshoot the minimum of the loss function, leading to oscillations and preventing the model from converging to a good solution.  The algorithm might jump around the loss landscape without settling on a minimum, resulting in poor performance.",
    },
    {
        tags: ["gradient"],
        number: 16,
        question: "What problem does vanishing gradient primarily affect?",
        options: [
            {
                letter: "a",
                answer: "Wide networks",
            },
            {
                letter: "b",
                answer: "Deep networks",
            },
            {
                letter: "c",
                answer: "Linear networks",
            },
            {
                letter: "d",
                answer: "Convolutional networks",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "Deep networks",
            },
        ],
        explanation:
            "The vanishing gradient problem primarily affects deep networks.  In deep networks with many layers, especially when using activation functions like sigmoid or tanh that saturate, the gradients can become extremely small during backpropagation. This makes it difficult or impossible for the weights in earlier layers to be updated effectively, hindering the learning process. Wide networks (a) are less susceptible because the gradient signal is distributed across more neurons. Linear networks (c) don't suffer from this because the gradient doesn't diminish with depth. While convolutional networks (d) are deep, they are also susceptible to vanishing gradients, especially if not carefully designed.",
    },
    {
        tags: ["training"],
        number: 17,
        question: "In backpropagation, which value propagates backward through the network?",
        options: [
            {
                letter: "a",
                answer: "Activations",
            },
            {
                letter: "b",
                answer: "Gradients of the loss function",
            },
            {
                letter: "c",
                answer: "Weight values",
            },
            {
                letter: "d",
                answer: "Bias terms",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "Gradients of the loss function",
            },
        ],
        explanation:
            "Backpropagation is the algorithm used to train neural networks. It works by calculating the gradient of the loss function with respect to the network's weights.  These gradients, representing the direction and magnitude of the error, are then propagated backward through the network to update the weights. Activations (a) are propagated forward. Weight values (c) and bias terms (d) are updated based on the calculated gradients.",
    },
    {
        tags: ["training"],
        number: 18,
        question: "Which of these are types of cost/loss functions?** *(Choose 2)",
        options: [
            {
                letter: "a",
                answer: "Mean Squared Error (MSE)",
            },
            {
                letter: "b",
                answer: "Cross-Entropy Loss",
            },
            {
                letter: "c",
                answer: "Batch Normalization",
            },
            {
                letter: "d",
                answer: "Gradient Descent",
            },
            {
                letter: "e",
                answer: "Softmax",
            },
        ],
        correct_answers: ["A", "B"],
        answers: [
            {
                letter: "a",
                answer: "Mean Squared Error (MSE)",
            },
            {
                letter: "b",
                answer: "Cross-Entropy Loss",
            },
        ],
        explanation:
            "Mean Squared Error (MSE) and Cross-Entropy Loss are both common cost/loss functions used in neural networks. MSE is typically used for regression tasks, measuring the average squared difference between predicted and actual values. Cross-entropy loss is commonly used for classification tasks, measuring the dissimilarity between the predicted probability distribution and the true distribution. Batch Normalization (c) is a normalization technique, not a loss function. Gradient Descent (d) is an optimization algorithm, and Softmax (e) is an activation function, not a loss function.",
    },
    {
        tags: ["training"],
        number: 19,
        question: "Scenario: If the learning rate is set to **0**, what will happen during training?",
        options: [
            {
                letter: "a",
                answer: "The model will not update its weights",
            },
            {
                letter: "b",
                answer: "The model will converge quickly",
            },
            {
                letter: "c",
                answer: "The cost function will increase exponentially",
            },
            {
                letter: "d",
                answer: "The model will overfit",
            },
        ],
        correct_answers: ["A"],
        answers: [
            {
                letter: "a",
                answer: "The model will not update its weights",
            },
        ],
        explanation:
            "The learning rate determines the step size taken during weight updates in the optimization process. If the learning rate is 0, the weight update equation becomes:  weights = weights - learning_rate * gradient.  Since the learning rate is 0, the weights remain unchanged, and the model will not learn.  The model will not converge (b), the cost function will not necessarily increase exponentially (c), and there will be no overfitting (d) because there is no learning.",
    },
    {
        tags: ["training"],
        number: 20,
        question: "Which two of the following optimizers are commonly used in neural networks?** *(Choose 2)",
        options: [
            {
                letter: "a",
                answer: "Adam",
            },
            {
                letter: "b",
                answer: "RMSProp",
            },
            {
                letter: "c",
                answer: "Softmax",
            },
            {
                letter: "d",
                answer: "Heaviside",
            },
            {
                letter: "e",
                answer: "Sigmoid",
            },
        ],
        correct_answers: ["A", "B"],
        answers: [
            {
                letter: "a",
                answer: "Adam",
            },
            {
                letter: "b",
                answer: "RMSProp",
            },
        ],
        explanation:
            "Adam (Adaptive Moment Estimation) and RMSProp (Root Mean Square Propagation) are both popular optimization algorithms used in training neural networks.  Adam combines ideas from RMSprop and momentum, adapting the learning rate for each parameter. RMSprop addresses the diminishing learning rates often encountered with AdaGrad. Softmax (c) is an activation function, and Heaviside (d) and Sigmoid (e) are activation functions, not optimizers.",
    },
    {
        tags: [],
        number: 21,
        question: "What is the purpose of a neural network's hidden layer?",
        options: [
            {
                letter: "a",
                answer: "Storing input data",
            },
            {
                letter: "b",
                answer: "Learning complex patterns and representations",
            },
            {
                letter: "c",
                answer: "Minimizing gradient values",
            },
            {
                letter: "d",
                answer: "Reducing the size of the network",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "Learning complex patterns and representations",
            },
        ],
        explanation:
            "Hidden layers in a neural network are crucial for learning complex, non-linear relationships in data.  They transform the input data through a series of weighted linear combinations and activation functions, creating increasingly abstract and informative representations at each layer.  This allows the network to model intricate patterns that a single-layer model (like linear regression) could not capture.",
    },
    {
        tags: ["training"],
        number: 22,
        question: 'What does the "weight initialization" step do?',
        options: [
            {
                letter: "a",
                answer: "Randomly assigns initial values to weights",
            },
            {
                letter: "b",
                answer: "Finds the best weights for each epoch",
            },
            {
                letter: "c",
                answer: "Reduces the cost function directly",
            },
            {
                letter: "d",
                answer: "Calculates gradients of the loss function",
            },
        ],
        correct_answers: ["A"],
        answers: [
            {
                letter: "a",
                answer: "Randomly assigns initial values to weights",
            },
        ],
        explanation:
            "Weight initialization is the process of assigning initial values to the weights of a neural network before training begins.  While various sophisticated initialization techniques exist (e.g., Xavier/Glorot initialization, He initialization), the fundamental purpose remains the same: to provide a starting point for the optimization algorithm.  These initial values are typically random, drawn from a specific distribution to avoid symmetry and promote efficient learning.  The choice of initialization strategy can significantly impact training speed and convergence.",
    },
    {
        tags: ["activation"],
        number: 23,
        question: "What happens if a neural network has **no activation function**?",
        options: [
            {
                letter: "a",
                answer: "It becomes non-linear",
            },
            {
                letter: "b",
                answer: "It behaves like a linear regression model",
            },
            {
                letter: "c",
                answer: "It converges faster",
            },
            {
                letter: "d",
                answer: "It overfits the training data",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "It behaves like a linear regression model",
            },
        ],
        explanation:
            "Without an activation function, the output of each neuron would be a simple linear combination of its inputs.  No matter how many layers you stack, the entire network would still perform only a linear transformation.  This severely limits its capacity to learn complex, non-linear patterns in the data, effectively reducing it to a linear regression model, regardless of its architecture.",
    },
    {
        tags: ["activation"],
        number: 24,
        question: "Which activation function is best suited for binary classification problems?",
        options: [
            {
                letter: "a",
                answer: "Sigmoid",
            },
            {
                letter: "b",
                answer: "Softmax",
            },
            {
                letter: "c",
                answer: "ReLU",
            },
            {
                letter: "d",
                answer: "Tanh",
            },
        ],
        correct_answers: ["A"],
        answers: [
            {
                letter: "a",
                answer: "Sigmoid",
            },
        ],
        explanation:
            "The sigmoid activation function outputs a value between 0 and 1, which can be directly interpreted as a probability.  In binary classification, we want to predict the probability of an instance belonging to one of two classes.  The sigmoid function's output range perfectly aligns with this requirement.  Softmax, while also producing probabilities, is typically used for multi-class classification problems.",
    },
    {
        tags: ["training"],
        number: 25,
        question: 'What does the "bias" term in a neuron enable?',
        options: [
            {
                letter: "a",
                answer: "It improves gradient flow",
            },
            {
                letter: "b",
                answer: "It allows the activation threshold to shift",
            },
            {
                letter: "c",
                answer: "It ensures vanishing gradients don't occur",
            },
            {
                letter: "d",
                answer: "It reduces model overfitting",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "It allows the activation threshold to shift",
            },
        ],
        explanation:
            "The bias term in a neuron adds a constant value to the weighted sum of inputs before the activation function is applied.  This effectively shifts the activation function's threshold.  Without a bias, the activation function would always be centered around zero, limiting the model's ability to learn and fit data effectively.  The bias allows the neuron to activate even when all input weights are zero, providing greater flexibility and expressiveness.",
    },
    {
        tags: ["training"],
        number: 26,
        question: "In backpropagation, the error is propagated:",
        options: [
            {
                letter: "a",
                answer: "Forward through the network",
            },
            {
                letter: "b",
                answer: "Backward from the output layer to the input layer",
            },
            {
                letter: "c",
                answer: "Randomly across all neurons",
            },
            {
                letter: "d",
                answer: "Within only the hidden layers",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "Backward from the output layer to the input layer",
            },
        ],
        explanation:
            "Backpropagation is the core algorithm for training neural networks.  It calculates the gradient of the loss function with respect to the network's weights. This gradient is then used to update the weights, minimizing the loss. The error is propagated backward, starting from the output layer and moving layer by layer towards the input layer.  Each layer's contribution to the overall error is calculated using the chain rule of calculus.",
    },
    {
        tags: [],
        number: 27,
        question: "Which component determines the size of weight updates during training?",
        options: [
            {
                letter: "a",
                answer: "Bias",
            },
            {
                letter: "b",
                answer: "Learning rate",
            },
            {
                letter: "c",
                answer: "Loss function",
            },
            {
                letter: "d",
                answer: "Activation function",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "Learning rate",
            },
        ],
        explanation:
            "The learning rate is a hyperparameter that controls the step size during weight updates in gradient descent-based optimization algorithms. A smaller learning rate leads to smaller weight updates, while a larger learning rate results in larger updates.  The learning rate directly influences the speed and stability of the training process.  An inappropriately large learning rate can cause the optimization process to overshoot the minimum, while a learning rate that is too small can lead to slow convergence.",
    },
    {
        tags: ["gradient", "training"],
        number: 28,
        question: "What kind of data does stochastic gradient descent (SGD) use to update weights?",
        options: [
            {
                letter: "a",
                answer: "The entire dataset",
            },
            {
                letter: "b",
                answer: "A single training example",
            },
            {
                letter: "c",
                answer: "A random batch of examples",
            },
            {
                letter: "d",
                answer: "Validation data only",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "A single training example",
            },
        ],
        explanation:
            "Stochastic Gradient Descent (SGD) updates the model's weights after processing a single training example.  Unlike batch gradient descent, which uses the entire dataset to compute the gradient, SGD uses only one data point at a time. This makes SGD computationally less expensive per iteration, but it can lead to noisy updates and slower convergence compared to batch gradient descent.  Mini-batch gradient descent offers a compromise, using a small batch of examples for each update.",
    },
    {
        tags: ["gradient"],
        number: 29,
        question: "How is batch gradient descent different from stochastic gradient descent?",
        options: [
            {
                letter: "a",
                answer: "It updates weights after seeing the full dataset",
            },
            {
                letter: "b",
                answer: "It computes gradients using a single data point",
            },
            {
                letter: "c",
                answer: "It doesn't use the learning rate",
            },
            {
                letter: "d",
                answer: "It works only for linear models",
            },
        ],
        correct_answers: ["A"],
        answers: [
            {
                letter: "a",
                answer: "It updates weights after seeing the full dataset",
            },
        ],
        explanation:
            "Batch gradient descent calculates the gradient of the loss function using the entire training dataset before updating the model's weights.  This contrasts with stochastic gradient descent, which updates weights after each training example.  Batch gradient descent provides a more accurate estimate of the gradient but is computationally expensive, especially for large datasets.  SGD, while less accurate per iteration, is often preferred for its efficiency and ability to escape local minima.",
    },
    {
        tags: ["training", "propagation"],
        number: 30,
        question: "What happens during the forward pass of backpropagation?",
        options: [
            {
                letter: "a",
                answer: "Gradients are calculated",
            },
            {
                letter: "b",
                answer: "Inputs are passed through the network to produce outputs",
            },
            {
                letter: "c",
                answer: "Weights are updated",
            },
            {
                letter: "d",
                answer: "Errors are minimized",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "Inputs are passed through the network to produce outputs",
            },
        ],
        explanation:
            "The forward pass is the first stage of backpropagation.  During the forward pass, the input data is fed through the neural network, layer by layer, until it reaches the output layer.  Each layer performs its computations (applying weights, biases, and activation functions) to produce its output, which then serves as the input for the next layer.  The output of the network is compared to the target output to calculate the loss, which is then used in the backward pass to compute gradients and update weights.",
    },
    {
        tags: ["training"],
        number: 31,
        question: "Which loss function is best suited for regression tasks?",
        options: [
            {
                letter: "a",
                answer: "Cross-Entropy Loss",
            },
            {
                letter: "b",
                answer: "Mean Squared Error (MSE)",
            },
            {
                letter: "c",
                answer: "Hinge Loss",
            },
            {
                letter: "d",
                answer: "Categorical Cross-Entropy",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "Mean Squared Error (MSE)",
            },
        ],
        explanation:
            "Mean Squared Error (MSE) is the most common loss function for regression tasks.  It calculates the average squared difference between the predicted and actual values.  Cross-entropy loss is used for classification problems. Hinge loss is used in Support Vector Machines (SVMs), and categorical cross-entropy is used for multi-class classification.",
    },
    {
        tags: ["training"],
        number: 32,
        question: "What does a smaller value of the cost function indicate?",
        options: [
            {
                letter: "a",
                answer: "Poor training performance",
            },
            {
                letter: "b",
                answer: "Model predictions are closer to actual values",
            },
            {
                letter: "c",
                answer: "Overfitting is occurring",
            },
            {
                letter: "d",
                answer: "Training hasn't started",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "Model predictions are closer to actual values",
            },
        ],
        explanation:
            "The cost function (or loss function) quantifies the error between the model's predictions and the true values. A smaller value indicates that the model's predictions are, on average, closer to the actual values, signifying better model performance.  Options A, C, and D describe situations where the cost function would likely be higher.",
    },
    {
        tags: [],
        number: 33,
        question: "What role does the optimizer play in neural network training?",
        options: [
            {
                letter: "a",
                answer: "It computes the cost function",
            },
            {
                letter: "b",
                answer: "It adjusts weights to minimize the loss function",
            },
            {
                letter: "c",
                answer: "It defines the network architecture",
            },
            {
                letter: "d",
                answer: "It normalizes the input features",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "It adjusts weights to minimize the loss function",
            },
        ],
        explanation:
            "The optimizer's role is to iteratively adjust the neural network's weights and biases to minimize the loss function.  It uses gradient descent or its variants (like Adam, RMSprop, etc.) to find the optimal parameters that reduce the error between predictions and actual values. Options A, C, and D describe other components of the neural network training process.",
    },
    {
        tags: ["gradient", "training"],
        number: 34,
        question: "Which two optimizers are variants of gradient descent?** *(Choose 2) e) Adagrad",
        options: [
            {
                letter: "a",
                answer: "Adam",
            },
            {
                letter: "b",
                answer: "RMSProp",
            },
            {
                letter: "c",
                answer: "BatchNorm",
            },
            {
                letter: "d",
                answer: "Gradient Boosting",
            },
        ],
        correct_answers: ["A", "B"],
        answers: [
            {
                letter: "a",
                answer: "Adam",
            },
            {
                letter: "b",
                answer: "RMSProp",
            },
        ],
        explanation:
            "Adam (Adaptive Moment Estimation) and RMSprop (Root Mean Square Propagation) are both adaptive optimization algorithms that are variants of gradient descent. They improve upon standard gradient descent by adapting the learning rate for each parameter.  Adagrad is also a variant, but the question asks for two. Gradient Boosting is an ensemble method, not an optimizer for neural networks.",
    },
    {
        tags: [],
        number: 35,
        question: "Scenario: During training, the model's loss decreases but its accuracy on validation data drops. What is likely happening?",
        options: [
            {
                letter: "a",
                answer: "Underfitting",
            },
            {
                letter: "b",
                answer: "Vanishing gradients",
            },
            {
                letter: "c",
                answer: "Overfitting",
            },
            {
                letter: "d",
                answer: "Poor weight initialization",
            },
        ],
        correct_answers: ["C"],
        answers: [
            {
                letter: "c",
                answer: "Overfitting",
            },
        ],
        explanation:
            "The scenario describes a classic case of overfitting. The model is learning the training data too well, resulting in a decrease in training loss. However, it is not generalizing well to unseen data (validation data), leading to a drop in validation accuracy.  Underfitting would show high loss on both training and validation sets. Vanishing gradients are a problem related to training dynamics, not directly indicated by this scenario. Poor weight initialization can affect training but doesn't specifically explain this pattern.",
    },
    {
        tags: ["gradient", "activation"],
        number: 36,
        question: "The vanishing gradient problem is most commonly associated with which activation function?",
        options: [
            {
                letter: "a",
                answer: "Sigmoid",
            },
            {
                letter: "b",
                answer: "ReLU",
            },
            {
                letter: "c",
                answer: "Softmax",
            },
            {
                letter: "d",
                answer: "Leaky ReLU",
            },
        ],
        correct_answers: ["A"],
        answers: [
            {
                letter: "a",
                answer: "Sigmoid",
            },
        ],
        explanation:
            "The sigmoid activation function, f(x) = 1 / (1 + exp(-x)), suffers from the vanishing gradient problem because its derivative is f'(x) = f(x)(1 - f(x)).  For large positive or negative inputs, the derivative approaches zero.  During backpropagation, this small derivative gets multiplied repeatedly across layers, leading to vanishing gradients and hindering learning in deeper networks.  ReLU and its variants are designed to mitigate this issue.",
    },
    {
        tags: ["activation"],
        number: 37,
        question: "What does the Tanh activation function output?",
        options: [
            {
                letter: "a",
                answer: "Values between 0 and 1",
            },
            {
                letter: "b",
                answer: "Values between -1 and 1",
            },
            {
                letter: "c",
                answer: "Values greater than 1",
            },
            {
                letter: "d",
                answer: "Binary outputs (0 or 1)",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "Values between -1 and 1",
            },
        ],
        explanation:
            "The hyperbolic tangent (Tanh) activation function, defined as tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)), outputs values in the range of -1 to 1. This is a key difference from the sigmoid function, which outputs values between 0 and 1.",
    },
    {
        tags: ["gradient", "activation"],
        number: 38,
        question: "Which activation function solves the vanishing gradient issue to some extent?",
        options: [
            {
                letter: "a",
                answer: "Sigmoid",
            },
            {
                letter: "b",
                answer: "Tanh",
            },
            {
                letter: "c",
                answer: "ReLU",
            },
            {
                letter: "d",
                answer: "Softmax",
            },
        ],
        correct_answers: ["C"],
        answers: [
            {
                letter: "c",
                answer: "ReLU",
            },
        ],
        explanation:
            "The Rectified Linear Unit (ReLU) activation function, f(x) = max(0, x), helps alleviate the vanishing gradient problem.  Its derivative is 1 for positive inputs and 0 for negative inputs.  The constant derivative of 1 for positive inputs prevents the gradients from shrinking to zero during backpropagation, allowing for more effective training of deeper networks compared to sigmoid or tanh.",
    },
    {
        tags: [],
        number: 39,
        question: "Scenario: A neural network consistently predicts the same class for all inputs. What is likely the problem?",
        options: [
            {
                letter: "a",
                answer: "Learning rate is too high",
            },
            {
                letter: "b",
                answer: "Model has saturated activation functions",
            },
            {
                letter: "c",
                answer: "Backpropagation is turned off",
            },
            {
                letter: "d",
                answer: "Loss function is incorrect",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "Model has saturated activation functions",
            },
        ],
        explanation:
            "If a neural network consistently predicts the same class, it suggests that the network's activations have saturated.  This means the activation functions are consistently outputting values close to their maximum or minimum, resulting in very small or zero gradients.  This prevents the network from learning and updating its weights effectively.  Sigmoid and tanh functions are particularly prone to saturation.",
    },
    {
        tags: ["gradient"],
        number: 40,
        question: "Which of the following techniques helps deal with vanishing gradients?",
        options: [
            {
                letter: "a",
                answer: "Using deeper networks",
            },
            {
                letter: "b",
                answer: "Weight initialization methods like He or Xavier",
            },
            {
                letter: "c",
                answer: "Removing activation functions",
            },
            {
                letter: "d",
                answer: "Reducing the learning rate",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "Weight initialization methods like He or Xavier",
            },
        ],
        explanation:
            "Weight initialization methods like He and Xavier initialization aim to prevent vanishing gradients by initializing the weights in a way that keeps the activations within a suitable range.  They address the problem of gradients becoming too small during backpropagation by ensuring that the signal doesn't diminish too rapidly as it propagates through the network's layers.  While other options might have some impact, they are not direct solutions to the vanishing gradient problem in the same way weight initialization is.",
    },
    {
        tags: [],
        number: 41,
        question: "What does a perceptron do in a neural network?",
        options: [
            {
                letter: "a",
                answer: "Multiplies input by weights and adds bias",
            },
            {
                letter: "b",
                answer: "Stores training data",
            },
            {
                letter: "c",
                answer: "Predicts the final output",
            },
            {
                letter: "d",
                answer: "Reduces overfitting",
            },
        ],
        correct_answers: ["A"],
        answers: [
            {
                letter: "a",
                answer: "Multiplies input by weights and adds bias",
            },
        ],
        explanation:
            "A perceptron, the fundamental building block of a neural network, performs a weighted sum of its inputs and adds a bias. This result is then passed through an activation function.",
    },
    {
        tags: ["activation"],
        number: 42,
        question: "What is the primary function of an activation function?",
        options: [
            {
                letter: "a",
                answer: "Calculate gradients during backpropagation",
            },
            {
                letter: "b",
                answer: "Introduce non-linearity into the network",
            },
            {
                letter: "c",
                answer: "Minimize the loss function",
            },
            {
                letter: "d",
                answer: "Adjust learning rate during training",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "Introduce non-linearity into the network",
            },
        ],
        explanation:
            "Without activation functions, a neural network would simply be a linear transformation of the input data, limiting its ability to learn complex patterns. Activation functions introduce non-linearity, enabling the network to approximate any continuous function.",
    },
    {
        tags: ["activation"],
        number: 43,
        question: "How is the output of a softmax activation function interpreted?",
        options: [
            {
                letter: "a",
                answer: "Probabilities for binary classification",
            },
            {
                letter: "b",
                answer: "Probabilities for multiple classes",
            },
            {
                letter: "c",
                answer: "Values between 0 and 1 for each neuron",
            },
            {
                letter: "d",
                answer: "Weighted inputs",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "Probabilities for multiple classes",
            },
        ],
        explanation:
            "The softmax function transforms a vector of arbitrary real numbers into a probability distribution.  Each element in the output vector represents the probability of belonging to a particular class, and the probabilities sum to 1.  This is crucial for multi-class classification problems.",
    },
    {
        tags: ["activation"],
        number: 44,
        question: "Which of the following is NOT a type of activation function?",
        options: [
            {
                letter: "a",
                answer: "Sigmoid",
            },
            {
                letter: "b",
                answer: "Tanh",
            },
            {
                letter: "c",
                answer: "Rectified Linear Unit (ReLU)",
            },
            {
                letter: "d",
                answer: "Exponential Linear Function",
            },
        ],
        correct_answers: ["D"],
        answers: [
            {
                letter: "d",
                answer: "Exponential Linear Function",
            },
        ],
        explanation:
            "While there are many variations and less common activation functions,  'Exponential Linear Function' isn't a standard or widely used activation function in the context of typical neural network architectures. Sigmoid, Tanh, and ReLU are all common activation functions.",
    },
    {
        tags: [],
        number: 45,
        question: "What does a neural network's depth refer to?",
        options: [
            {
                letter: "a",
                answer: "Number of hidden layers",
            },
            {
                letter: "b",
                answer: "Number of input features",
            },
            {
                letter: "c",
                answer: "Number of output neurons",
            },
            {
                letter: "d",
                answer: "Size of each layer",
            },
        ],
        correct_answers: ["A"],
        answers: [
            {
                letter: "a",
                answer: "Number of hidden layers",
            },
        ],
        explanation:
            "The depth of a neural network refers to the number of hidden layers it contains.  A deeper network (more layers) can, in theory, learn more complex representations of the data, although this comes with increased computational cost and the risk of vanishing/exploding gradients.",
    },
    {
        tags: ["training"],
        number: 46,
        question: "Which of the following elements is required for backpropagation?",
        options: [
            {
                letter: "a",
                answer: "Forward pass output",
            },
            {
                letter: "b",
                answer: "Gradients of the loss function",
            },
            {
                letter: "c",
                answer: "Learning rate",
            },
            {
                letter: "d",
                answer: "All of the above",
            },
        ],
        correct_answers: ["D"],
        answers: [
            {
                letter: "d",
                answer: "All of the above",
            },
        ],
        explanation:
            "Backpropagation requires the forward pass output to compute the loss, the gradients of the loss function to determine the direction of weight updates, and a learning rate to control the step size of these updates.  All three are essential components.",
    },
    {
        tags: ["gradient"],
        number: 47,
        question: "What causes the vanishing gradient problem?",
        options: [
            {
                letter: "a",
                answer: "High learning rate",
            },
            {
                letter: "b",
                answer: "Large number of layers with small gradients",
            },
            {
                letter: "c",
                answer: "Poor loss function design",
            },
            {
                letter: "d",
                answer: "Use of the ReLU activation",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "Large number of layers with small gradients",
            },
        ],
        explanation:
            "The vanishing gradient problem occurs when gradients become extremely small during backpropagation, especially in deep networks. This is primarily due to repeated multiplication of gradients less than 1 across many layers, leading to near-zero gradients that hinder effective weight updates in earlier layers.  The use of activation functions with saturated regions (like sigmoid) exacerbates this issue.",
    },
    {
        tags: ["gradient", "training"],
        number: 48,
        question: "During backpropagation, gradients flow from:",
        options: [
            {
                letter: "a",
                answer: "Input layer to output layer",
            },
            {
                letter: "b",
                answer: "Output layer back to input layer",
            },
            {
                letter: "c",
                answer: "Hidden layers to the loss function",
            },
            {
                letter: "d",
                answer: "Neurons to biases",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "Output layer back to input layer",
            },
        ],
        explanation:
            "Backpropagation is a chain rule-based algorithm that calculates gradients of the loss function with respect to the network's weights.  This process starts at the output layer, where the loss is computed, and propagates backward through the network, layer by layer, until it reaches the input layer.  Gradients are calculated and used to update weights at each layer.",
    },
    {
        tags: ["gradient"],
        number: 49,
        question: "Which two methods can help mitigate the vanishing gradient problem?** *(Choose 2) e) Increasing the loss value artificially",
        options: [
            {
                letter: "a",
                answer: "Using Tanh activation function",
            },
            {
                letter: "b",
                answer: "Using ReLU activation function",
            },
            {
                letter: "c",
                answer: "Applying batch normalization",
            },
            {
                letter: "d",
                answer: "Using smaller learning rates",
            },
        ],
        correct_answers: ["B", "C"],
        answers: [
            {
                letter: "b",
                answer: "Using ReLU activation function",
            },
            {
                letter: "c",
                answer: "Applying batch normalization",
            },
        ],
        explanation:
            "The vanishing gradient problem can be mitigated by using activation functions like ReLU, which avoids the saturation problem of sigmoid and tanh functions.  Batch normalization helps stabilize the learning process by normalizing the activations of each layer, preventing gradients from becoming too small.  Increasing the loss value artificially is not a standard or effective method for addressing the vanishing gradient problem.",
    },
    {
        tags: ["gradient", "training"],
        number: 50,
        question: "What role does the chain rule play in backpropagation?",
        options: [
            {
                letter: "a",
                answer: "Calculates gradients by combining partial derivatives",
            },
            {
                letter: "b",
                answer: "Reduces the size of the network",
            },
            {
                letter: "c",
                answer: "Finds the activation function",
            },
            {
                letter: "d",
                answer: "Optimizes the weight updates",
            },
        ],
        correct_answers: ["A"],
        answers: [
            {
                letter: "a",
                answer: "Calculates gradients by combining partial derivatives",
            },
        ],
        explanation:
            "The chain rule is fundamental to backpropagation.  It allows the calculation of the gradient of the loss function with respect to each weight by breaking down the complex gradient calculation into a series of simpler partial derivatives, each computed layer by layer. This efficient computation is crucial for updating the network's weights.",
    },
    {
        tags: ["training"],
        number: 51,
        question: "Which loss function is ideal for binary classification tasks?",
        options: [
            {
                letter: "a",
                answer: "Mean Squared Error",
            },
            {
                letter: "b",
                answer: "Hinge Loss",
            },
            {
                letter: "c",
                answer: "Binary Cross-Entropy",
            },
            {
                letter: "d",
                answer: "KL Divergence",
            },
        ],
        correct_answers: ["C"],
        answers: [
            {
                letter: "c",
                answer: "Binary Cross-Entropy",
            },
        ],
        explanation:
            "Binary cross-entropy is the ideal loss function for binary classification because it directly measures the difference between the predicted probability and the true binary label (0 or 1).  Mean Squared Error (MSE) can be used, but it's less suitable as it's not designed for probabilities and can be less sensitive to misclassifications compared to cross-entropy. Hinge loss is typically used in Support Vector Machines (SVMs), not directly in neural networks for binary classification. KL Divergence measures the difference between two probability distributions, which is not directly applicable in the same way as cross-entropy for binary classification.",
    },
    {
        tags: ["gradient"],
        number: 52,
        question: 'What does the term "gradient" refer to in gradient descent?',
        options: [
            {
                letter: "a",
                answer: "The difference between predicted and actual output",
            },
            {
                letter: "b",
                answer: "The slope of the loss function",
            },
            {
                letter: "c",
                answer: "The activation function value",
            },
            {
                letter: "d",
                answer: "The sum of input weights",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "The slope of the loss function",
            },
        ],
        explanation:
            "In gradient descent, the gradient represents the direction and magnitude of the steepest ascent of the loss function at a particular point in the parameter space.  The negative of the gradient is used to update the model parameters, moving them in the direction of the steepest descent (reducing the loss).",
    },
    {
        tags: [],
        number: 53,
        question: "Scenario: A neural network shows a loss that fluctuates sharply during training. Which strategy can stabilize it?",
        options: [
            {
                letter: "a",
                answer: "Increase batch size",
            },
            {
                letter: "b",
                answer: "Use a sigmoid activation function",
            },
            {
                letter: "c",
                answer: "Reduce the depth of the network",
            },
            {
                letter: "d",
                answer: "Increase learning rate",
            },
        ],
        correct_answers: ["A"],
        answers: [
            {
                letter: "a",
                answer: "Increase batch size",
            },
        ],
        explanation:
            "Sharp fluctuations in the loss during training often indicate high variance in the gradient estimates. Increasing the batch size reduces the variance of the gradient estimate, leading to smoother updates and a more stable training process.  A larger batch size provides a more accurate representation of the overall gradient, reducing the noise inherent in smaller batches.  While reducing network depth or changing activation functions *might* help, increasing the batch size is a more direct and common approach to stabilize training.",
    },
    {
        tags: ["gradient"],
        number: 54,
        question: "How does the Adam optimizer differ from SGD?",
        options: [
            {
                letter: "a",
                answer: "It computes gradients faster",
            },
            {
                letter: "b",
                answer: "It adapts learning rates for each parameter",
            },
            {
                letter: "c",
                answer: "It works only for linear networks",
            },
            {
                letter: "d",
                answer: "It ignores the cost function",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "It adapts learning rates for each parameter",
            },
        ],
        explanation:
            "Adam (Adaptive Moment Estimation) is an optimization algorithm that adapts the learning rate for each parameter individually.  It does this by maintaining a running average of past gradients (momentum) and their squares (variance). This allows Adam to efficiently navigate the loss landscape, often converging faster than standard Stochastic Gradient Descent (SGD), which uses a single learning rate for all parameters.  Adam does not compute gradients faster inherently; it's the adaptive learning rate that contributes to faster convergence.",
    },
    {
        tags: ["training"],
        number: 55,
        question: "What happens when the learning rate is set too low?",
        options: [
            {
                letter: "a",
                answer: "The model fails to converge",
            },
            {
                letter: "b",
                answer: "The model converges very slowly",
            },
            {
                letter: "c",
                answer: "The loss function increases indefinitely",
            },
            {
                letter: "d",
                answer: "The gradients vanish",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "The model converges very slowly",
            },
        ],
        explanation:
            "A learning rate that is too low results in very small updates to the model's parameters during each iteration. This means the model will take a very long time to converge to a good solution.  While extremely low learning rates might theoretically prevent convergence in some edge cases (e.g., getting stuck in a local minimum), the most common and practical consequence is slow convergence.",
    },
    {
        tags: ["activation"],
        number: 56,
        question: "Which activation function can output a value of exactly zero?",
        options: [
            {
                letter: "a",
                answer: "Sigmoid",
            },
            {
                letter: "b",
                answer: "Tanh",
            },
            {
                letter: "c",
                answer: "ReLU",
            },
            {
                letter: "d",
                answer: "Softmax",
            },
        ],
        correct_answers: ["C"],
        answers: [
            {
                letter: "c",
                answer: "ReLU",
            },
        ],
        explanation:
            "The ReLU (Rectified Linear Unit) activation function is defined as f(x) = max(0, x).  Therefore, for any input x \u2264 0, the output is exactly zero. Sigmoid, Tanh, and Softmax functions, on the other hand, always output values within a specific range (0 to 1 for sigmoid, -1 to 1 for Tanh, and a probability distribution summing to 1 for softmax), never reaching exactly zero except in limiting cases.",
    },
    {
        tags: ["activation"],
        number: 57,
        question: "What does the leaky ReLU activation function address?",
        options: [
            {
                letter: "a",
                answer: "Vanishing gradients",
            },
            {
                letter: "b",
                answer: "Exploding gradients",
            },
            {
                letter: "c",
                answer: "Dead neurons",
            },
            {
                letter: "d",
                answer: "Overfitting",
            },
        ],
        correct_answers: ["C"],
        answers: [
            {
                letter: "c",
                answer: "Dead neurons",
            },
        ],
        explanation:
            "The Leaky ReLU addresses the \"dying ReLU\" problem.  Standard ReLU units output zero for negative inputs, and if a neuron consistently receives negative inputs during training, its gradient will always be zero, preventing it from learning (a 'dead neuron'). Leaky ReLU introduces a small slope for negative inputs, allowing the neuron to still learn even with negative activations, thus mitigating the dead neuron problem.  While it can indirectly help with vanishing gradients, its primary purpose is to address the dead neuron issue.",
    },
    {
        tags: ["activation"],
        number: 58,
        question: "In which layer do you typically use the softmax activation function?",
        options: [
            {
                letter: "a",
                answer: "Input layer",
            },
            {
                letter: "b",
                answer: "Hidden layer",
            },
            {
                letter: "c",
                answer: "Output layer for multi-class classification",
            },
            {
                letter: "d",
                answer: "Any layer with large inputs",
            },
        ],
        correct_answers: ["C"],
        answers: [
            {
                letter: "c",
                answer: "Output layer for multi-class classification",
            },
        ],
        explanation:
            "The softmax function is typically used in the output layer of a neural network designed for multi-class classification. It transforms the raw output scores from the network into a probability distribution over the different classes.  Each output neuron represents a class, and the softmax function ensures that the outputs sum to 1, representing the probability of the input belonging to each class. Using softmax in other layers would not be meaningful in the context of multi-class classification.",
    },
    {
        tags: ["training"],
        number: 59,
        question: "Which of the following scenarios leads to overfitting in a neural network?",
        options: [
            {
                letter: "a",
                answer: "Training for too many epochs without regularization",
            },
            {
                letter: "b",
                answer: "Using a large learning rate",
            },
            {
                letter: "c",
                answer: "Insufficient number of hidden layers",
            },
            {
                letter: "d",
                answer: "High batch size",
            },
        ],
        correct_answers: ["A"],
        answers: [
            {
                letter: "a",
                answer: "Training for too many epochs without regularization",
            },
        ],
        explanation:
            "Overfitting occurs when a model learns the training data too well, including its noise and outliers, resulting in poor generalization to unseen data. Training for too many epochs without regularization techniques (like dropout, weight decay, or early stopping) allows the model to memorize the training data, leading to overfitting. A large learning rate can lead to instability and prevent convergence, but not necessarily overfitting. Insufficient hidden layers might lead to underfitting, and a high batch size generally improves generalization and reduces overfitting.",
    },
    {
        tags: [],
        number: 60,
        question: "Scenario: A neural network performs well on training data but poorly on validation data. Which of these approaches can improve the situation?",
        options: [
            {
                letter: "a",
                answer: "Use dropout regularization",
            },
            {
                letter: "b",
                answer: "Reduce model depth",
            },
            {
                letter: "c",
                answer: "Train with fewer epochs",
            },
            {
                letter: "d",
                answer: "Use a larger learning rate",
            },
        ],
        correct_answers: ["A"],
        answers: [
            {
                letter: "a",
                answer: "Use dropout regularization",
            },
        ],
        explanation:
            "The scenario describes overfitting: good performance on training data but poor performance on validation data. Dropout is a regularization technique that randomly ignores neurons during training, preventing the network from relying too heavily on any single neuron or feature. This forces the network to learn more robust and generalizable features, improving its performance on unseen data. Reducing model depth or training with fewer epochs might help, but dropout is a more targeted approach to address overfitting. Increasing the learning rate is unlikely to help and could worsen the situation.",
    },
    {
        tags: [],
        number: 61,
        question: "Which of the following is NOT a hyperparameter in neural networks?",
        options: [
            {
                letter: "a",
                answer: "Learning rate",
            },
            {
                letter: "b",
                answer: "Number of hidden layers",
            },
            {
                letter: "c",
                answer: "Weight values",
            },
            {
                letter: "d",
                answer: "Batch size",
            },
        ],
        correct_answers: ["C"],
        answers: [
            {
                letter: "c",
                answer: "Weight values",
            },
        ],
        explanation:
            "Hyperparameters are parameters that are set *before* the training process begins, and they control the learning process itself.  Learning rate, number of hidden layers, and batch size are all examples of hyperparameters. Weight values, on the other hand, are parameters *learned* during the training process through backpropagation and gradient descent. They are adjusted iteratively to minimize the loss function.",
    },
    {
        tags: ["training"],
        number: 62,
        question: "What is the role of bias in a neuron?",
        options: [
            {
                letter: "a",
                answer: "Control the activation function output",
            },
            {
                letter: "b",
                answer: "Shift the activation function curve",
            },
            {
                letter: "c",
                answer: "Multiply inputs by weights",
            },
            {
                letter: "d",
                answer: "Reduce overfitting",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "Shift the activation function curve",
            },
        ],
        explanation:
            "The bias term in a neuron is added to the weighted sum of inputs *before* the activation function is applied.  This addition shifts the activation function's curve along the x-axis.  Without a bias, the activation function would always pass through the origin (0,0), limiting its ability to model diverse data patterns.  The bias allows the neuron to activate even when all inputs are zero.",
    },
    {
        tags: [],
        number: 63,
        question: "Which layer processes the raw input data in a neural network?",
        options: [
            {
                letter: "a",
                answer: "Output layer",
            },
            {
                letter: "b",
                answer: "Hidden layer",
            },
            {
                letter: "c",
                answer: "Input layer",
            },
            {
                letter: "d",
                answer: "Fully connected layer",
            },
        ],
        correct_answers: ["C"],
        answers: [
            {
                letter: "c",
                answer: "Input layer",
            },
        ],
        explanation:
            "The input layer is the first layer in a neural network. It receives the raw input data, which is then processed and passed to subsequent layers.  The input layer typically has one neuron for each feature in the input data.  The input layer doesn't perform any computation; it simply feeds the data to the next layer.",
    },
    {
        tags: ["training"],
        number: 64,
        question: "Which technique helps improve generalization in a neural network?",
        options: [
            {
                letter: "a",
                answer: "Increasing the learning rate",
            },
            {
                letter: "b",
                answer: "Adding more training data",
            },
            {
                letter: "c",
                answer: "Removing the activation function",
            },
            {
                letter: "d",
                answer: "Using fewer layers",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "Adding more training data",
            },
        ],
        explanation:
            "Generalization refers to the ability of a model to perform well on unseen data.  Adding more training data helps improve generalization because it exposes the model to a wider range of examples, reducing the risk of overfitting to the specific characteristics of the training set.  Increasing the learning rate can lead to instability and prevent convergence. Removing the activation function removes the non-linearity, limiting the model's capacity. Using fewer layers might underfit the data.",
    },
    {
        tags: ["activation"],
        number: 65,
        question: "What happens if the activation function is removed?",
        options: [
            {
                letter: "a",
                answer: "The neural network becomes linear",
            },
            {
                letter: "b",
                answer: "The loss decreases rapidly",
            },
            {
                letter: "c",
                answer: "Gradients vanish",
            },
            {
                letter: "d",
                answer: "Training speed increases",
            },
        ],
        correct_answers: ["A"],
        answers: [
            {
                letter: "a",
                answer: "The neural network becomes linear",
            },
        ],
        explanation:
            "Activation functions introduce non-linearity into the neural network.  Without an activation function, the output of each neuron would be a simple linear combination of its inputs.  This means that the entire network would effectively be a linear model, regardless of the number of layers.  A linear model has limited capacity to learn complex patterns in data.",
    },
    {
        tags: ["activation"],
        number: 66,
        question: "Which activation function is most commonly used in deep learning for hidden layers?",
        options: [
            {
                letter: "a",
                answer: "Sigmoid",
            },
            {
                letter: "b",
                answer: "ReLU",
            },
            {
                letter: "c",
                answer: "Softmax",
            },
            {
                letter: "d",
                answer: "Tanh",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "ReLU",
            },
        ],
        explanation:
            "While various activation functions are used in deep learning, ReLU (Rectified Linear Unit) is the most prevalent choice for hidden layers.  Its simplicity (f(x) = max(0, x)) and efficiency in mitigating the vanishing gradient problem compared to sigmoid or tanh make it a popular choice.  Softmax is typically reserved for the output layer in multi-class classification problems.",
    },
    {
        tags: ["activation"],
        number: 67,
        question: "What is the output range of the Tanh activation function?",
        options: [
            {
                letter: "a",
                answer: "0 to 1",
            },
            {
                letter: "b",
                answer: "-1 to 1",
            },
            {
                letter: "c",
                answer: "0 to \u00e2\u02c6\u017e",
            },
            {
                letter: "d",
                answer: "-\u00e2\u02c6\u017e to +\u00e2\u02c6\u017e",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "-1 to 1",
            },
        ],
        explanation: "The hyperbolic tangent function (tanh) outputs values in the range of -1 to 1. This is a key difference from the sigmoid function, which outputs values between 0 and 1.",
    },
    {
        tags: ["activation"],
        number: 68,
        question: "The sigmoid activation function is suitable for:",
        options: [
            {
                letter: "a",
                answer: "Linear regression problems",
            },
            {
                letter: "b",
                answer: "Binary classification tasks",
            },
            {
                letter: "c",
                answer: "Multi-class classification",
            },
            {
                letter: "d",
                answer: "Minimizing overfitting",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "Binary classification tasks",
            },
        ],
        explanation:
            "The sigmoid function outputs a value between 0 and 1, which can be interpreted as a probability. This makes it suitable for binary classification problems where the output represents the probability of belonging to one of two classes.  While it can be used in other contexts, its limitations (vanishing gradients and non-zero centered output) make it less ideal for other tasks.",
    },
    {
        tags: ["gradient", "activation"],
        number: 69,
        question: "Which two activation functions are prone to the vanishing gradient problem?** *(Choose 2) e) Softmax",
        options: [
            {
                letter: "a",
                answer: "ReLU",
            },
            {
                letter: "b",
                answer: "Sigmoid",
            },
            {
                letter: "c",
                answer: "Tanh",
            },
            {
                letter: "d",
                answer: "Leaky ReLU",
            },
        ],
        correct_answers: ["B", "C"],
        answers: [
            {
                letter: "b",
                answer: "Sigmoid",
            },
            {
                letter: "c",
                answer: "Tanh",
            },
        ],
        explanation:
            "Both sigmoid and tanh activation functions suffer from the vanishing gradient problem, especially in deep networks.  Their derivatives approach zero as the input values move towards their extreme ends, hindering the backpropagation process and making it difficult to train deep networks effectively. ReLU and Leaky ReLU are designed to mitigate this issue.",
    },
    {
        tags: ["activation"],
        number: 70,
        question: "What is the advantage of using the Leaky ReLU over ReLU?",
        options: [
            {
                letter: "a",
                answer: "It avoids dead neurons",
            },
            {
                letter: "b",
                answer: "It increases gradient flow",
            },
            {
                letter: "c",
                answer: "It performs better on small datasets",
            },
            {
                letter: "d",
                answer: "It always outputs positive values",
            },
        ],
        correct_answers: ["A"],
        answers: [
            {
                letter: "a",
                answer: "It avoids dead neurons",
            },
        ],
        explanation:
            "The ReLU activation function has a derivative of 0 for negative inputs, leading to 'dead neurons' where the weights are not updated during backpropagation. Leaky ReLU addresses this by introducing a small, non-zero slope for negative inputs (e.g., f(x) = max(0.01x, x)), allowing for some gradient flow even when the input is negative. This helps prevent the vanishing gradient problem and keeps neurons active.",
    },
    {
        tags: ["training"],
        number: 71,
        question: "During backpropagation, what is propagated backwards through the network?",
        options: [
            {
                letter: "a",
                answer: "Input features",
            },
            {
                letter: "b",
                answer: "Gradients of loss with respect to weights",
            },
            {
                letter: "c",
                answer: "Predictions",
            },
            {
                letter: "d",
                answer: "Bias values",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "Gradients of loss with respect to weights",
            },
        ],
        explanation:
            "Backpropagation is the core algorithm for training neural networks.  It calculates the gradient of the loss function with respect to the network's weights. This gradient indicates the direction and magnitude of the weight adjustments needed to reduce the loss.  The gradients, not the input features, predictions, or bias values themselves, are propagated backward through the network to update the weights.",
    },
    {
        tags: ["training"],
        number: 72,
        question: "Which step occurs FIRST in backpropagation?",
        options: [
            {
                letter: "a",
                answer: "Update weights using gradients",
            },
            {
                letter: "b",
                answer: "Compute loss function",
            },
            {
                letter: "c",
                answer: "Calculate gradient using chain rule",
            },
            {
                letter: "d",
                answer: "Apply activation function",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "Compute loss function",
            },
        ],
        explanation:
            "Before calculating gradients and updating weights, the loss function must first be computed. The loss function quantifies the difference between the network's predictions and the actual target values.  The gradient calculation (using the chain rule) and weight updates follow the computation of the loss.",
    },
    {
        tags: ["gradient"],
        number: 73,
        question: "If gradients explode during training, what is a potential solution?",
        options: [
            {
                letter: "a",
                answer: "Use gradient clipping",
            },
            {
                letter: "b",
                answer: "Increase learning rate",
            },
            {
                letter: "c",
                answer: "Add more hidden layers",
            },
            {
                letter: "d",
                answer: "Remove activation functions",
            },
        ],
        correct_answers: ["A"],
        answers: [
            {
                letter: "a",
                answer: "Use gradient clipping",
            },
        ],
        explanation:
            "Exploding gradients occur when the gradients become excessively large during training, leading to instability and potentially NaN values. Gradient clipping is a common solution. It limits the magnitude of the gradients to a predefined threshold, preventing them from becoming too large and causing instability. Increasing the learning rate would exacerbate the problem, adding more layers doesn't directly address the gradient explosion, and removing activation functions would likely severely impair network performance.",
    },
    {
        tags: ["gradient", "training"],
        number: 74,
        question: "Which parameter determines how much weights are updated in each step of gradient descent?",
        options: [
            {
                letter: "a",
                answer: "Loss function",
            },
            {
                letter: "b",
                answer: "Batch size",
            },
            {
                letter: "c",
                answer: "Learning rate",
            },
            {
                letter: "d",
                answer: "Activation function",
            },
        ],
        correct_answers: ["C"],
        answers: [
            {
                letter: "c",
                answer: "Learning rate",
            },
        ],
        explanation:
            "The learning rate is a hyperparameter that controls the step size during gradient descent. It determines how much the weights are updated in each iteration based on the calculated gradients. A smaller learning rate leads to smaller weight updates, while a larger learning rate leads to larger updates. The loss function measures the error, batch size affects the gradient calculation, and the activation function introduces non-linearity but doesn't directly control the weight update magnitude.",
    },
    {
        tags: ["gradient"],
        number: 75,
        question: "What causes the exploding gradient problem?",
        options: [
            {
                letter: "a",
                answer: "Small gradients in deep networks",
            },
            {
                letter: "b",
                answer: "High magnitude weight updates",
            },
            {
                letter: "c",
                answer: "Poor activation function choice",
            },
            {
                letter: "d",
                answer: "Using SGD instead of Adam",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "High magnitude weight updates",
            },
        ],
        explanation:
            "The exploding gradient problem arises from very large weight updates during backpropagation.  These large updates can lead to instability in the training process, causing the weights to diverge and the network to fail to converge. While poor activation function choice or the choice of optimizer can contribute, the root cause is the high magnitude of the weight updates, resulting in excessively large gradients.",
    },
    {
        tags: ["training"],
        number: 76,
        question: "What type of loss function is best suited for multi-class classification problems?",
        options: [
            {
                letter: "a",
                answer: "Mean Absolute Error",
            },
            {
                letter: "b",
                answer: "Mean Squared Error",
            },
            {
                letter: "c",
                answer: "Cross-Entropy Loss",
            },
            {
                letter: "d",
                answer: "Hinge Loss",
            },
        ],
        correct_answers: ["C"],
        answers: [
            {
                letter: "c",
                answer: "Cross-Entropy Loss",
            },
        ],
        explanation:
            "Cross-entropy loss is the most suitable loss function for multi-class classification problems.  It measures the difference between the predicted probability distribution and the true distribution.  Mean Absolute Error (MAE) and Mean Squared Error (MSE) are better suited for regression problems. Hinge loss is typically used in support vector machines (SVMs), not commonly in deep learning multi-class classification.",
    },
    {
        tags: ["gradient"],
        number: 77,
        question: 'What does the term "stochastic" mean in stochastic gradient descent?',
        options: [
            {
                letter: "a",
                answer: "It uses the entire dataset at each step",
            },
            {
                letter: "b",
                answer: "It updates weights for a random subset of data",
            },
            {
                letter: "c",
                answer: "It reduces randomness in training",
            },
            {
                letter: "d",
                answer: "It guarantees convergence",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "It updates weights for a random subset of data",
            },
        ],
        explanation:
            "Stochastic Gradient Descent (SGD) updates the model's weights based on the gradient calculated from a small random sample (a mini-batch) of the training data. This contrasts with batch gradient descent, which uses the entire dataset for each update, and is computationally more expensive.  The randomness introduced by using mini-batches helps escape local minima and speeds up the training process.",
    },
    {
        tags: ["training"],
        number: 78,
        question: "Which optimizer adapts the learning rate for each parameter individually?",
        options: [
            {
                letter: "a",
                answer: "SGD",
            },
            {
                letter: "b",
                answer: "Adam",
            },
            {
                letter: "c",
                answer: "Adagrad",
            },
            {
                letter: "d",
                answer: "RMSProp",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "Adam",
            },
        ],
        explanation:
            "Adam (Adaptive Moment Estimation) is an optimizer that adapts the learning rate for each parameter individually. It uses both the first and second moments of the gradients to dynamically adjust the learning rate.  While Adagrad and RMSProp also adapt learning rates, Adam is generally considered more effective and widely used due to its combination of adaptive learning rates and momentum.",
    },
    {
        tags: ["training"],
        number: 79,
        question: "How does batch size affect the training process?",
        options: [
            {
                letter: "a",
                answer: "Smaller batches lead to smoother weight updates",
            },
            {
                letter: "b",
                answer: "Larger batches increase computational cost",
            },
            {
                letter: "c",
                answer: "Smaller batches add noise to gradients",
            },
            {
                letter: "d",
                answer: "Larger batches stabilize gradients",
            },
        ],
        correct_answers: ["B", "C", "D"],
        answers: [
            {
                letter: "b",
                answer: "Larger batches increase computational cost",
            },
            {
                letter: "c",
                answer: "Smaller batches add noise to gradients",
            },
            {
                letter: "d",
                answer: "Larger batches stabilize gradients",
            },
        ],
        explanation:
            "Batch size significantly impacts training. Larger batches reduce the noise in the gradient estimate, leading to more stable updates but increasing computational cost per iteration. Smaller batches introduce more noise, potentially helping escape local minima but making the training process less stable.  Option A is partially true; smaller batches lead to more frequent updates, but not necessarily smoother in the sense of a consistently decreasing loss curve. The noise introduced can lead to oscillations.",
    },
    {
        tags: [],
        number: 80,
        question: "Scenario: During training, your loss value stops decreasing. What is the FIRST step you should try?",
        options: [
            {
                letter: "a",
                answer: "Reduce the learning rate",
            },
            {
                letter: "b",
                answer: "Increase the number of hidden layers",
            },
            {
                letter: "c",
                answer: "Remove dropout regularization",
            },
            {
                letter: "d",
                answer: "Increase batch size",
            },
        ],
        correct_answers: ["A"],
        answers: [
            {
                letter: "a",
                answer: "Reduce the learning rate",
            },
        ],
        explanation:
            "If the loss stops decreasing during training, the first troubleshooting step should be to reduce the learning rate. A learning rate that is too high can cause the optimization algorithm to overshoot the minimum, preventing convergence.  Increasing the number of hidden layers, removing dropout (which is a regularization technique), or increasing the batch size are more significant architectural or hyperparameter changes that should be considered only after simpler adjustments like reducing the learning rate have been explored.",
    },
    {
        tags: [],
        number: 81,
        question: "Which of the following components is updated during training?",
        options: [
            {
                letter: "a",
                answer: "Loss function",
            },
            {
                letter: "b",
                answer: "Input values",
            },
            {
                letter: "c",
                answer: "Weights and biases",
            },
            {
                letter: "d",
                answer: "Activation function",
            },
        ],
        correct_answers: ["C"],
        answers: [
            {
                letter: "c",
                answer: "Weights and biases",
            },
        ],
        explanation:
            "During training, the weights and biases of the neural network are adjusted to minimize the loss function.  The loss function itself doesn't change during a single training step (though its value does). Input values are the data fed into the network and remain constant during a single training iteration. The activation function is a fixed mathematical operation.",
    },
    {
        tags: [],
        number: 82,
        question: "What does a neuron compute in a neural network?",
        options: [
            {
                letter: "a",
                answer: "The average of input values",
            },
            {
                letter: "b",
                answer: "A weighted sum of inputs plus bias",
            },
            {
                letter: "c",
                answer: "The loss value",
            },
            {
                letter: "d",
                answer: "The gradient of weights",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "A weighted sum of inputs plus bias",
            },
        ],
        explanation:
            "A neuron in a neural network performs a weighted sum of its inputs, adds a bias term, and then applies an activation function to produce its output. This is the fundamental computation unit of a neural network.",
    },
    {
        tags: ["training"],
        number: 83,
        question: 'What does the term "epoch" mean in neural network training?',
        options: [
            {
                letter: "a",
                answer: "A single update of all weights",
            },
            {
                letter: "b",
                answer: "One forward pass of the network",
            },
            {
                letter: "c",
                answer: "One backward pass of the network",
            },
            {
                letter: "d",
                answer: "One complete pass through the entire training dataset",
            },
        ],
        correct_answers: ["D"],
        answers: [
            {
                letter: "d",
                answer: "One complete pass through the entire training dataset",
            },
        ],
        explanation:
            "An epoch represents one complete cycle of training where the entire training dataset is passed through the neural network once.  Each epoch typically involves multiple batches and updates to the network's weights and biases.",
    },
    {
        tags: [],
        number: 84,
        question: "Which factor determines the number of outputs from a neural network?",
        options: [
            {
                letter: "a",
                answer: "The activation function used",
            },
            {
                letter: "b",
                answer: "The number of neurons in the output layer",
            },
            {
                letter: "c",
                answer: "The number of hidden layers",
            },
            {
                letter: "d",
                answer: "The batch size",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "The number of neurons in the output layer",
            },
        ],
        explanation:
            "The number of outputs from a neural network is directly determined by the number of neurons in the output layer. Each neuron in the output layer produces one output value. The activation function influences the range and distribution of the outputs but not the number of them.  Hidden layers and batch size affect the training process but not the final number of outputs.",
    },
    {
        tags: [],
        number: 85,
        question: "In a fully connected layer, each neuron is connected to:",
        options: [
            {
                letter: "a",
                answer: "Neurons in the previous layer only",
            },
            {
                letter: "b",
                answer: "Neurons in the next layer only",
            },
            {
                letter: "c",
                answer: "All neurons in the previous layer",
            },
            {
                letter: "d",
                answer: "All neurons in the next layer",
            },
        ],
        correct_answers: ["C"],
        answers: [
            {
                letter: "c",
                answer: "All neurons in the previous layer",
            },
        ],
        explanation:
            "In a fully connected layer (also known as a dense layer), every neuron in that layer is connected to every neuron in the preceding layer.  This creates a dense network of connections, hence the name.  There are no connections skipped.",
    },
    {
        tags: ["activation"],
        number: 86,
        question: "Which of the following is a characteristic of the ReLU activation function?",
        options: [
            {
                letter: "a",
                answer: "Outputs range from -1 to 1",
            },
            {
                letter: "b",
                answer: "It is non-linear",
            },
            {
                letter: "c",
                answer: "It solves the vanishing gradient problem entirely",
            },
            {
                letter: "d",
                answer: "It is differentiable at all points",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "It is non-linear",
            },
        ],
        explanation:
            "ReLU (Rectified Linear Unit) is a non-linear activation function.  Option a is incorrect because ReLU outputs values from 0 to infinity. Option c is incorrect because while ReLU alleviates the vanishing gradient problem to some extent by avoiding saturation for positive inputs, it doesn't solve it entirely. Option d is incorrect because ReLU is not differentiable at x=0; it has a subgradient at that point.",
    },
    {
        tags: ["activation"],
        number: 87,
        question: "What output does the softmax activation function produce?",
        options: [
            {
                letter: "a",
                answer: "Binary values (0 or 1)",
            },
            {
                letter: "b",
                answer: "Probabilities that sum to 1",
            },
            {
                letter: "c",
                answer: "Weighted inputs",
            },
            {
                letter: "d",
                answer: "Negative values only",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "Probabilities that sum to 1",
            },
        ],
        explanation:
            "The softmax function transforms a vector of arbitrary real numbers into a probability distribution.  The output is a vector where each element represents a probability, and the sum of all elements is always 1.",
    },
    {
        tags: ["activation"],
        number: 88,
        question: "Why is the sigmoid function rarely used in hidden layers?",
        options: [
            {
                letter: "a",
                answer: "It is computationally expensive",
            },
            {
                letter: "b",
                answer: "It causes exploding gradients",
            },
            {
                letter: "c",
                answer: "It is prone to vanishing gradients",
            },
            {
                letter: "d",
                answer: "It cannot handle negative inputs",
            },
        ],
        correct_answers: ["C"],
        answers: [
            {
                letter: "c",
                answer: "It is prone to vanishing gradients",
            },
        ],
        explanation:
            "The sigmoid function's output saturates at both ends (approaching 0 and 1), leading to very small gradients during backpropagation.  This is the vanishing gradient problem, which hinders learning, especially in deep networks. While it's not computationally expensive (option a), and it doesn't inherently cause exploding gradients (option b), the vanishing gradient problem is its primary drawback in hidden layers.",
    },
    {
        tags: ["activation"],
        number: 89,
        question: "Which activation function would you use in a binary classification problem?",
        options: [
            {
                letter: "a",
                answer: "Softmax",
            },
            {
                letter: "b",
                answer: "ReLU",
            },
            {
                letter: "c",
                answer: "Sigmoid",
            },
            {
                letter: "d",
                answer: "Tanh",
            },
        ],
        correct_answers: ["C"],
        answers: [
            {
                letter: "c",
                answer: "Sigmoid",
            },
        ],
        explanation:
            "The sigmoid function outputs a value between 0 and 1, which can be interpreted as a probability. This makes it suitable for binary classification problems where the output represents the probability of belonging to one of the two classes. Softmax is used for multi-class classification.",
    },
    {
        tags: ["activation"],
        number: 90,
        question: "The Tanh activation function is useful because:",
        options: [
            {
                letter: "a",
                answer: "It outputs only positive values",
            },
            {
                letter: "b",
                answer: "It centers the output around zero",
            },
            {
                letter: "c",
                answer: "It reduces gradient vanishing",
            },
            {
                letter: "d",
                answer: "It improves training speed",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "It centers the output around zero",
            },
        ],
        explanation:
            "The tanh (hyperbolic tangent) function outputs values between -1 and 1, centered around 0.  This centering can sometimes lead to faster convergence during training compared to sigmoid, which outputs values between 0 and 1. While it helps mitigate the vanishing gradient problem to some extent (option c), its primary advantage is the zero-centered output.",
    },
    {
        tags: ["training"],
        number: 91,
        question: "What mathematical rule is central to backpropagation?",
        options: [
            {
                letter: "a",
                answer: "Gradient descent rule",
            },
            {
                letter: "b",
                answer: "Chain rule of differentiation",
            },
            {
                letter: "c",
                answer: "Linear algebra transformations",
            },
            {
                letter: "d",
                answer: "Cross-entropy loss",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "Chain rule of differentiation",
            },
        ],
        explanation:
            "Backpropagation relies heavily on the chain rule to calculate gradients of the loss function with respect to the weights in each layer.  The chain rule allows us to efficiently compute these gradients by breaking down the complex composite function into smaller, manageable derivatives.  Gradient descent is the optimization algorithm that *uses* these gradients, but the chain rule is the fundamental mathematical principle enabling the calculation.",
    },
    {
        tags: ["training"],
        number: 92,
        question: "In backpropagation, which of the following occurs LAST?",
        options: [
            {
                letter: "a",
                answer: "Compute gradients using chain rule",
            },
            {
                letter: "b",
                answer: "Update weights using gradients",
            },
            {
                letter: "c",
                answer: "Calculate the loss function",
            },
            {
                letter: "d",
                answer: "Forward propagate input through the network",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "Update weights using gradients",
            },
        ],
        explanation:
            "The backpropagation algorithm follows these steps: 1) Forward propagation of the input; 2) Calculation of the loss function; 3) Computation of gradients using the chain rule; 4) Update of weights using the calculated gradients (e.g., via gradient descent).  Weight updates happen *after* the gradients have been computed.",
    },
    {
        tags: ["gradient"],
        number: 93,
        question: "Which scenario indicates vanishing gradients?",
        options: [
            {
                letter: "a",
                answer: "Gradients become very small in deeper layers",
            },
            {
                letter: "b",
                answer: "Gradients increase uncontrollably",
            },
            {
                letter: "c",
                answer: "Loss function fluctuates during training",
            },
            {
                letter: "d",
                answer: "Output layer neurons fail to activate",
            },
        ],
        correct_answers: ["A"],
        answers: [
            {
                letter: "a",
                answer: "Gradients become very small in deeper layers",
            },
        ],
        explanation:
            "Vanishing gradients refer to the phenomenon where gradients become increasingly small as they propagate backward through many layers of a deep neural network. This makes it difficult for the network to learn effectively, especially in the earlier layers.  The other options describe different issues: exploding gradients (b), unstable training (c), and issues with activation functions (d).",
    },
    {
        tags: ["gradient", "training"],
        number: 94,
        question: "How is the learning rate related to gradient descent?",
        options: [
            {
                letter: "a",
                answer: "It determines the direction of weight updates",
            },
            {
                letter: "b",
                answer: "It determines the size of weight updates",
            },
            {
                letter: "c",
                answer: "It prevents overfitting",
            },
            {
                letter: "d",
                answer: "It directly controls the loss function",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "It determines the size of weight updates",
            },
        ],
        explanation:
            "The learning rate is a hyperparameter that controls the step size during gradient descent. A smaller learning rate leads to smaller weight updates, while a larger learning rate results in larger weight updates. The direction of the weight update is determined by the gradient itself, not the learning rate. The learning rate indirectly influences the loss function by affecting the speed of convergence.",
    },
    {
        tags: ["training"],
        number: 95,
        question: "If your training loss is high and validation loss is low, what is likely happening?",
        options: [
            {
                letter: "a",
                answer: "Overfitting",
            },
            {
                letter: "b",
                answer: "Underfitting",
            },
            {
                letter: "c",
                answer: "Gradient explosion",
            },
            {
                letter: "d",
                answer: "Poor initialization",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "Underfitting",
            },
        ],
        explanation:
            "High training loss indicates the model is not learning the training data well.  Low validation loss, however, suggests the model generalizes well to unseen data. This discrepancy points to underfitting: the model is too simple to capture the complexity of the training data.  Overfitting would show low training loss and high validation loss.",
    },
    {
        tags: ["gradient"],
        number: 96,
        question: "Which optimizer is known for combining momentum and RMSProp?",
        options: [
            {
                letter: "a",
                answer: "Adam",
            },
            {
                letter: "b",
                answer: "Adagrad",
            },
            {
                letter: "c",
                answer: "SGD",
            },
            {
                letter: "d",
                answer: "Nesterov",
            },
        ],
        correct_answers: ["A"],
        answers: [
            {
                letter: "a",
                answer: "Adam",
            },
        ],
        explanation:
            "Adam (Adaptive Moment Estimation) optimizer is explicitly designed to combine the benefits of both momentum and RMSprop.  Momentum helps accelerate SGD in relevant directions and dampens oscillations, while RMSprop addresses the varying learning rates for different parameters by using a moving average of squared gradients. Adam effectively incorporates both these mechanisms.",
    },
    {
        tags: ["training"],
        number: 97,
        question: "What does the loss function measure in neural network training?",
        options: [
            {
                letter: "a",
                answer: "Accuracy of predictions",
            },
            {
                letter: "b",
                answer: "Difference between predicted and actual values",
            },
            {
                letter: "c",
                answer: "Gradient flow during training",
            },
            {
                letter: "d",
                answer: "Weight adjustments",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "Difference between predicted and actual values",
            },
        ],
        explanation:
            "The loss function quantifies the discrepancy between the neural network's predictions and the true target values.  Minimizing this loss is the primary goal during training.  Options A, C, and D describe other aspects of training but not the direct purpose of the loss function.",
    },
    {
        tags: ["training"],
        number: 98,
        question: "Which loss function is most appropriate for regression tasks?",
        options: [
            {
                letter: "a",
                answer: "Binary Cross-Entropy",
            },
            {
                letter: "b",
                answer: "Categorical Cross-Entropy",
            },
            {
                letter: "c",
                answer: "Mean Squared Error",
            },
            {
                letter: "d",
                answer: "Hinge Loss",
            },
        ],
        correct_answers: ["C"],
        answers: [
            {
                letter: "c",
                answer: "Mean Squared Error",
            },
        ],
        explanation:
            "Mean Squared Error (MSE) is a common loss function for regression problems. It measures the average squared difference between predicted and actual values.  Binary and Categorical Cross-Entropy are used for classification tasks, while Hinge Loss is typically used in Support Vector Machines (SVMs).",
    },
    {
        tags: ["gradient"],
        number: 99,
        question: "What is the primary purpose of mini-batch gradient descent?",
        options: [
            {
                letter: "a",
                answer: "To reduce computation time per update",
            },
            {
                letter: "b",
                answer: "To stabilize weight initialization",
            },
            {
                letter: "c",
                answer: "To improve activation function output",
            },
            {
                letter: "d",
                answer: "To reduce dataset size",
            },
        ],
        correct_answers: ["A"],
        answers: [
            {
                letter: "a",
                answer: "To reduce computation time per update",
            },
        ],
        explanation:
            "Mini-batch gradient descent processes the training data in smaller batches (mini-batches) instead of the entire dataset at once. This significantly reduces the computational cost of each gradient update, making training more efficient, especially for large datasets.  It also introduces a degree of stochasticity which can help escape local minima.",
    },
    {
        tags: [],
        number: 100,
        question: "Scenario: Your neural network training is progressing slowly. What should you try FIRST?",
        options: [
            {
                letter: "a",
                answer: "Increase the number of hidden layers",
            },
            {
                letter: "b",
                answer: "Increase the learning rate",
            },
            {
                letter: "c",
                answer: "Add dropout regularization",
            },
            {
                letter: "d",
                answer: "Switch activation functions",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "Increase the learning rate",
            },
        ],
        explanation:
            "Slow training often indicates that the learning rate is too small. Increasing the learning rate allows the optimizer to take larger steps in the parameter space, potentially leading to faster convergence. However, increasing the learning rate too much can lead to divergence.  The other options (increasing layers, adding dropout, or changing activation functions) are more significant architectural changes that should be considered only after simpler solutions like adjusting the learning rate have been explored.  It's crucial to monitor the training process closely after any adjustment.",
    },
    {
        tags: ["training"],
        number: 101,
        question: "What role does bias play in a neuron's computation?",
        options: [
            {
                letter: "a",
                answer: "It scales the inputs",
            },
            {
                letter: "b",
                answer: "It shifts the activation function output",
            },
            {
                letter: "c",
                answer: "It adds non-linearity to the inputs",
            },
            {
                letter: "d",
                answer: "It normalizes the gradients",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "It shifts the activation function output",
            },
        ],
        explanation:
            "The bias term in a neuron is added to the weighted sum of inputs before the activation function is applied.  This addition acts as a shift, changing the activation function's output.  It doesn't scale inputs (a), add non-linearity (c) directly (the activation function handles that), or normalize gradients (d).",
    },
    {
        tags: [],
        number: 102,
        question: "Which of the following is NOT a type of neural network?",
        options: [
            {
                letter: "a",
                answer: "Convolutional Neural Network (CNN)",
            },
            {
                letter: "b",
                answer: "Recurrent Neural Network (RNN)",
            },
            {
                letter: "c",
                answer: "Decision Tree Network",
            },
            {
                letter: "d",
                answer: "Fully Connected Network",
            },
        ],
        correct_answers: ["C"],
        answers: [
            {
                letter: "c",
                answer: "Decision Tree Network",
            },
        ],
        explanation:
            "Decision trees are a type of machine learning algorithm, but they are not a type of neural network.  CNNs, RNNs, and fully connected networks are all architectures within the realm of neural networks.",
    },
    {
        tags: [],
        number: 103,
        question: "The hidden layers of a neural network primarily perform:",
        options: [
            {
                letter: "a",
                answer: "Data input preprocessing",
            },
            {
                letter: "b",
                answer: "Feature extraction and transformation",
            },
            {
                letter: "c",
                answer: "Activation function initialization",
            },
            {
                letter: "d",
                answer: "Gradient computation",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "Feature extraction and transformation",
            },
        ],
        explanation:
            "Hidden layers learn increasingly complex representations of the input data.  They extract features and transform them into higher-level abstractions, which are then used by subsequent layers or the output layer for the final prediction.  Preprocessing (a) happens before the network, activation initialization (c) is a setup step, and gradient computation (d) is part of the training process, not the primary function of hidden layers.",
    },
    {
        tags: [],
        number: 104,
        question: "How does increasing the number of hidden layers affect the network?",
        options: [
            {
                letter: "a",
                answer: "It reduces computational cost",
            },
            {
                letter: "b",
                answer: "It makes the network capable of learning more complex representations",
            },
            {
                letter: "c",
                answer: "It decreases the risk of overfitting",
            },
            {
                letter: "d",
                answer: "It eliminates the need for activation functions",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "It makes the network capable of learning more complex representations",
            },
        ],
        explanation:
            "Deeper networks (more hidden layers) can learn more complex and hierarchical representations of the data.  However, this comes at the cost of increased computational complexity (a) and a higher risk of overfitting (c), which needs to be managed through techniques like regularization, dropout, or early stopping.  Activation functions are still necessary (d).",
    },
    {
        tags: ["activation"],
        number: 105,
        question: "What is the output of a softmax function in a network with 3 output neurons?",
        options: [
            {
                letter: "a",
                answer: "A single binary value (0 or 1)",
            },
            {
                letter: "b",
                answer: "Three probabilities that sum to 1",
            },
            {
                letter: "c",
                answer: "Weighted sum of inputs",
            },
            {
                letter: "d",
                answer: "Gradients of input values",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "Three probabilities that sum to 1",
            },
        ],
        explanation:
            "The softmax function transforms a vector of arbitrary real numbers into a probability distribution.  With 3 output neurons, it produces three probabilities, each representing the likelihood of belonging to a specific class.  These probabilities always sum to 1, representing a complete probability distribution over the three classes.",
    },
    {
        tags: ["activation"],
        number: 106,
        question: "Which activation function is defined as \\( \\text{ReLU}(x) = \\max(0, x) \\)?",
        options: [
            {
                letter: "a",
                answer: "Sigmoid",
            },
            {
                letter: "b",
                answer: "Tanh",
            },
            {
                letter: "c",
                answer: "Leaky ReLU",
            },
            {
                letter: "d",
                answer: "Rectified Linear Unit (ReLU)",
            },
        ],
        correct_answers: ["D"],
        answers: [
            {
                letter: "d",
                answer: "Rectified Linear Unit (ReLU)",
            },
        ],
        explanation:
            "The definition \\\\( \\\\text{ReLU}(x) = \\\\max(0, x) \\\\) directly corresponds to the Rectified Linear Unit activation function.  ReLU outputs the input if it's positive and 0 otherwise.",
    },
    {
        tags: ["activation"],
        number: 107,
        question: "What issue does Leaky ReLU aim to solve?",
        options: [
            {
                letter: "a",
                answer: "Exploding gradients",
            },
            {
                letter: "b",
                answer: "Vanishing gradients",
            },
            {
                letter: "c",
                answer: "Computational inefficiency",
            },
            {
                letter: "d",
                answer: "Overfitting",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "Vanishing gradients",
            },
        ],
        explanation:
            "Leaky ReLU addresses the vanishing gradient problem.  The vanishing gradient problem occurs when gradients become very small during backpropagation in deep networks, hindering learning in earlier layers.  Leaky ReLU introduces a small, non-zero slope for negative inputs, preventing the gradient from completely vanishing.",
    },
    {
        tags: ["activation"],
        number: 108,
        question: "In which situation is the softmax activation function typically used?",
        options: [
            {
                letter: "a",
                answer: "Regression tasks",
            },
            {
                letter: "b",
                answer: "Binary classification",
            },
            {
                letter: "c",
                answer: "Multi-class classification",
            },
            {
                letter: "d",
                answer: "Feature scaling",
            },
        ],
        correct_answers: ["C"],
        answers: [
            {
                letter: "c",
                answer: "Multi-class classification",
            },
        ],
        explanation:
            "The softmax function is typically used in multi-class classification problems. It transforms a vector of arbitrary real numbers into a probability distribution over multiple classes, ensuring the outputs sum to 1.  This allows for the interpretation of the outputs as probabilities of belonging to each class.",
    },
    {
        tags: ["activation"],
        number: 109,
        question: "What is the output range of the sigmoid activation function?",
        options: [
            {
                letter: "a",
                answer: "\\(-1\\) to \\(1\\)",
            },
            {
                letter: "b",
                answer: "\\(0\\) to \\(1\\)",
            },
            {
                letter: "c",
                answer: "\\(-\\infty\\) to \\(\\infty\\)",
            },
            {
                letter: "d",
                answer: "\\(0\\) to \\(0.5\\)",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "0 to 1",
            },
        ],
        explanation:
            "The sigmoid function's output range is between 0 and 1 (exclusive).  This makes it suitable for binary classification problems where the output can be interpreted as a probability.",
    },
    {
        tags: ["activation"],
        number: 110,
        question: "Why is the ReLU activation function preferred in many deep networks?",
        options: [
            {
                letter: "a",
                answer: "It outputs smooth gradients",
            },
            {
                letter: "b",
                answer: "It prevents gradient vanishing issues",
            },
            {
                letter: "c",
                answer: "It handles binary outputs well",
            },
            {
                letter: "d",
                answer: "It is computationally expensive",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "It prevents gradient vanishing issues",
            },
        ],
        explanation:
            "ReLU is preferred in many deep networks because it helps mitigate the vanishing gradient problem.  The non-zero slope for positive inputs ensures that gradients don't shrink to zero during backpropagation, allowing for effective training of deeper layers. While ReLU can suffer from the 'dying ReLU' problem (neurons becoming inactive), this is often less severe than the vanishing gradient problem.",
    },
    {
        tags: ["training"],
        number: 111,
        question: "Which loss function is best suited for binary classification problems?",
        options: [
            {
                letter: "a",
                answer: "Mean Squared Error",
            },
            {
                letter: "b",
                answer: "Hinge Loss",
            },
            {
                letter: "c",
                answer: "Binary Cross-Entropy",
            },
            {
                letter: "d",
                answer: "Softmax Loss",
            },
        ],
        correct_answers: ["C"],
        answers: [
            {
                letter: "c",
                answer: "Binary Cross-Entropy",
            },
        ],
        explanation:
            "Binary cross-entropy is the most suitable loss function for binary classification problems.  It measures the dissimilarity between the predicted probability distribution and the true binary labels.  Mean Squared Error (MSE) can be used, but it's less effective than cross-entropy for classification tasks because it doesn't directly model probabilities. Hinge loss is typically used in Support Vector Machines (SVMs), not directly in neural networks for binary classification. Softmax loss is for multi-class classification, not binary.",
    },
    {
        tags: ["training"],
        number: 112,
        question: "What happens when the learning rate is set too high?",
        options: [
            {
                letter: "a",
                answer: "The model converges faster",
            },
            {
                letter: "b",
                answer: "The gradients explode, and training becomes unstable",
            },
            {
                letter: "c",
                answer: "The weights remain constant",
            },
            {
                letter: "d",
                answer: "The loss function minimizes smoothly",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "The gradients explode, and training becomes unstable",
            },
        ],
        explanation:
            "A learning rate that is too high can lead to unstable training.  The updates to the weights become too large, causing the model to overshoot the optimal weights and potentially diverge.  This phenomenon is often referred to as gradient explosion.  The model will not converge smoothly; instead, the loss function might oscillate wildly or even increase over time.",
    },
    {
        tags: ["gradient"],
        number: 113,
        question: "The gradient descent algorithm minimizes which quantity?",
        options: [
            {
                letter: "a",
                answer: "Accuracy",
            },
            {
                letter: "b",
                answer: "Number of epochs",
            },
            {
                letter: "c",
                answer: "Loss function value",
            },
            {
                letter: "d",
                answer: "Activation function outputs",
            },
        ],
        correct_answers: ["C"],
        answers: [
            {
                letter: "c",
                answer: "Loss function value",
            },
        ],
        explanation:
            "The core goal of gradient descent is to minimize the value of the loss function.  The algorithm iteratively updates the model's parameters (weights and biases) in the direction of the negative gradient of the loss function, thus reducing the loss with each step. Accuracy is a metric, not directly minimized by gradient descent. Epochs are iterations over the entire dataset, and activation function outputs are intermediate values in the network.",
    },
    {
        tags: ["gradient"],
        number: 114,
        question: "How does stochastic gradient descent (SGD) differ from batch gradient descent?",
        options: [
            {
                letter: "a",
                answer: "SGD updates weights using the entire dataset",
            },
            {
                letter: "b",
                answer: "SGD updates weights after every single data point",
            },
            {
                letter: "c",
                answer: "SGD updates weights less frequently",
            },
            {
                letter: "d",
                answer: "SGD requires no learning rate",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "SGD updates weights after every single data point",
            },
        ],
        explanation:
            "Stochastic Gradient Descent (SGD) updates the model's weights after processing each individual data point.  Batch gradient descent, on the other hand, calculates the gradient using the entire training dataset before updating the weights.  This difference significantly impacts computational cost and convergence behavior. SGD is faster per iteration but can be noisier in its convergence.",
    },
    {
        tags: ["gradient"],
        number: 115,
        question: "Why is mini-batch gradient descent widely used?",
        options: [
            {
                letter: "a",
                answer: "It requires no gradients",
            },
            {
                letter: "b",
                answer: "It balances computational efficiency and convergence stability",
            },
            {
                letter: "c",
                answer: "It avoids using activation functions",
            },
            {
                letter: "d",
                answer: "It eliminates the need for backpropagation",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "It balances computational efficiency and convergence stability",
            },
        ],
        explanation:
            "Mini-batch gradient descent uses a small subset (mini-batch) of the training data to compute the gradient and update the weights. This approach offers a good compromise between the computational efficiency of SGD and the smoother convergence of batch gradient descent.  It reduces the noise inherent in SGD while remaining computationally feasible for large datasets.  It doesn't avoid activation functions or eliminate backpropagation; these are fundamental components of neural network training.",
    },
    {
        tags: ["training"],
        number: 116,
        question: "What is the purpose of backpropagation in a neural network?",
        options: [
            {
                letter: "a",
                answer: "To initialize weights and biases",
            },
            {
                letter: "b",
                answer: "To propagate input values to the output layer",
            },
            {
                letter: "c",
                answer: "To calculate and distribute gradients for updating weights",
            },
            {
                letter: "d",
                answer: "To apply activation functions",
            },
        ],
        correct_answers: ["C"],
        answers: [
            {
                letter: "c",
                answer: "To calculate and distribute gradients for updating weights",
            },
        ],
        explanation:
            "Backpropagation is the core algorithm for training neural networks. It uses the chain rule of calculus to compute the gradient of the loss function with respect to the network's weights and biases. These gradients indicate the direction and magnitude of weight adjustments needed to reduce the loss and improve the network's performance. Options A, B, and D describe other processes in neural networks but not the primary function of backpropagation.",
    },
    {
        tags: ["gradient", "training"],
        number: 117,
        question: "During backpropagation, the gradients are computed for:",
        options: [
            {
                letter: "a",
                answer: "The weights and biases",
            },
            {
                letter: "b",
                answer: "The activation function outputs",
            },
            {
                letter: "c",
                answer: "The input values only",
            },
            {
                letter: "d",
                answer: "The loss function input",
            },
        ],
        correct_answers: ["A"],
        answers: [
            {
                letter: "a",
                answer: "The weights and biases",
            },
        ],
        explanation:
            "Backpropagation calculates the gradients of the loss function with respect to the network's weights and biases.  These gradients are crucial for updating the weights and biases using an optimization algorithm like gradient descent, thereby adjusting the network to better fit the training data.  The gradients for activation function outputs are implicitly calculated as part of the chain rule application to obtain the weight and bias gradients.",
    },
    {
        tags: ["gradient", "training"],
        number: 118,
        question: "What does the chain rule help compute during backpropagation?",
        options: [
            {
                letter: "a",
                answer: "Forward propagation outputs",
            },
            {
                letter: "b",
                answer: "Gradients of intermediate and output layers",
            },
            {
                letter: "c",
                answer: "Bias initialization",
            },
            {
                letter: "d",
                answer: "Loss function values",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "Gradients of intermediate and output layers",
            },
        ],
        explanation:
            "The chain rule is essential in backpropagation because it allows for the efficient computation of gradients at each layer.  It breaks down the complex gradient calculation into smaller, manageable steps, propagating the gradient from the output layer back to the input layer.  This enables the calculation of gradients for both intermediate and output layers, guiding the weight updates.",
    },
    {
        tags: ["gradient"],
        number: 119,
        question: "Which of the following is a symptom of gradient explosion?",
        options: [
            {
                letter: "a",
                answer: "Gradients decrease to near-zero values",
            },
            {
                letter: "b",
                answer: "Weights are updated to excessively large values",
            },
            {
                letter: "c",
                answer: "Loss becomes constant during training",
            },
            {
                letter: "d",
                answer: "The network stops learning after a few epochs",
            },
        ],
        correct_answers: ["B"],
        answers: [
            {
                letter: "b",
                answer: "Weights are updated to excessively large values",
            },
        ],
        explanation:
            "Gradient explosion occurs when the gradients during backpropagation become excessively large, leading to unstable weight updates.  These large updates can cause the weights to explode to very large values, making the network unstable and preventing convergence. Options A, C, and D describe other potential training issues, but not the defining characteristic of gradient explosion.",
    },
    {
        tags: ["gradient"],
        number: 120,
        question: "To combat vanishing gradients, which action is most helpful?",
        options: [
            {
                letter: "a",
                answer: "Use the sigmoid activation function",
            },
            {
                letter: "b",
                answer: "Apply a smaller learning rate",
            },
            {
                letter: "c",
                answer: "Use activation functions like ReLU or Leaky ReLU",
            },
            {
                letter: "d",
                answer: "Increase the number of output neurons",
            },
        ],
        correct_answers: ["C"],
        answers: [
            {
                letter: "c",
                answer: "Use activation functions like ReLU or Leaky ReLU",
            },
        ],
        explanation:
            "Vanishing gradients occur when gradients during backpropagation become very small, hindering the learning process, especially in deep networks.  Sigmoid and tanh activation functions suffer from this problem due to their saturating nature. ReLU (Rectified Linear Unit) and Leaky ReLU address this by having a non-saturating region, allowing for more stable gradient propagation during backpropagation.  Options A and B are not effective solutions; a smaller learning rate might slow down the process but not solve the root cause, and the number of output neurons is irrelevant to the vanishing gradient problem.",
    },
]
