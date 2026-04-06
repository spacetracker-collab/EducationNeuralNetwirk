import numpy as np

np.random.seed(42)

# Activation functions
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

# Loss function (Mean Squared Error)
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def mse_derivative(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.size


# Layer class
class Layer:
    def __init__(self, input_size, output_size, name):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.bias = np.zeros((1, output_size))
        self.name = name

    def forward(self, x):
        self.input = x
        self.z = np.dot(x, self.weights) + self.bias
        self.output = leaky_relu(self.z)
        return self.output

    def backward(self, grad_output, learning_rate):
        dz = grad_output * leaky_relu_derivative(self.z)
        dw = np.dot(self.input.T, dz)
        db = np.sum(dz, axis=0, keepdims=True)

        grad_input = np.dot(dz, self.weights.T)

        # Update weights
        self.weights -= learning_rate * dw
        self.bias -= learning_rate * db

        return grad_input


# Education Neural Network
class EducationNN:
    def __init__(self):
        self.layers = [
            Layer(5, 10, "Primary"),
            Layer(10, 20, "Middle School"),
            Layer(20, 30, "High School"),
            Layer(30, 25, "Undergraduate"),
            Layer(25, 15, "Masters"),
            Layer(15, 10, "PhD"),
            Layer(10, 5, "Postdoc")
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_grad, learning_rate):
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad, learning_rate)


# Simulated Input (Raw Student)
def generate_student():
    return np.array([[0.2, 0.3, 0.5, 0.4, 0.6]])


# Ideal Output (Expert Target)
def generate_target():
    return np.array([[0.8, 0.7, 0.9, 0.6, 0.75]])


if __name__ == "__main__":
    model = EducationNN()

    student_input = generate_student()
    target_output = generate_target()

    epochs = 1000
    lr = 0.01

    print("Training Education Neural Network...\n")

    for epoch in range(epochs):
        # Forward pass
        output = model.forward(student_input)

        # Loss
        loss = mse_loss(output, target_output)

        # Backward pass
        grad = mse_derivative(output, target_output)
        model.backward(grad, lr)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")

    print("\nFinal Evaluation:")
    final_output = model.forward(student_input)

    print("Input (Raw Student):", student_input)
    print("Target (Ideal Expert):", target_output)
    print("Final Output (Learned Expert):", final_output)
