import numpy as np

np.random.seed(42)

# -----------------------------
# Activation Functions
# -----------------------------
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# -----------------------------
# Loss Functions (Multi-task)
# -----------------------------
def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def mse_grad(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.size


# -----------------------------
# Attention Mechanism
# -----------------------------
class Attention:
    def __init__(self, size):
        self.weights = np.random.randn(size, size) * 0.1

    def forward(self, x):
        scores = np.dot(x, self.weights)
        self.attn = sigmoid(scores)
        return x * self.attn

    def backward(self, grad, x, lr):
        dscores = grad * x * self.attn * (1 - self.attn)
        dw = np.dot(x.T, dscores)
        self.weights -= lr * dw
        return grad * self.attn


# -----------------------------
# Layer Definition
# -----------------------------
class Layer:
    def __init__(self, in_size, out_size, name):
        self.w = np.random.randn(in_size, out_size) * 0.1
        self.b = np.zeros((1, out_size))
        self.name = name

    def forward(self, x):
        self.input = x
        self.z = np.dot(x, self.w) + self.b
        self.out = leaky_relu(self.z)
        return self.out

    def backward(self, grad, lr):
        dz = grad * leaky_relu_derivative(self.z)
        dw = np.dot(self.input.T, dz)
        db = np.sum(dz, axis=0, keepdims=True)

        grad_input = np.dot(dz, self.w.T)

        self.w -= lr * dw
        self.b -= lr * db

        return grad_input


# -----------------------------
# Education Neural Network
# -----------------------------
class EducationNN:
    def __init__(self):
        self.layers = [
            Layer(5, 16, "Primary"),
            Layer(16, 32, "Middle"),
            Layer(32, 32, "HighSchool"),
            Layer(32, 24, "UG"),
            Layer(24, 16, "Masters"),
            Layer(16, 10, "PhD"),
            Layer(10, 5, "Postdoc")
        ]
        self.attention = Attention(5)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        self.pre_attn = x
        x = self.attention.forward(x)
        return x

    def backward(self, grad, lr):
        grad = self.attention.backward(grad, self.pre_attn, lr)

        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr)


# -----------------------------
# Curriculum Learning Targets
# -----------------------------
def curriculum_targets(stage):
    # Gradually increasing expectations
    base = np.array([[0.2, 0.2, 0.2, 0.2, 0.2]])
    target = np.array([[0.8, 0.7, 0.9, 0.6, 0.75]])
    return base + (target - base) * stage


# -----------------------------
# Multi-Student Simulation
# -----------------------------
def generate_students(n=5):
    return np.random.rand(n, 5)


# -----------------------------
# Training
# -----------------------------
if __name__ == "__main__":
    model = EducationNN()

    students = generate_students(10)
    lr = 0.01
    epochs = 1000

    print("Training Multi-Student Education System...\n")

    for epoch in range(epochs):
        total_loss = 0

        # Curriculum stage grows over time
        stage = min(1.0, epoch / 500)
        target = curriculum_targets(stage)

        for student in students:
            x = student.reshape(1, -1)

            # Forward
            output = model.forward(x)

            # Multi-loss (each dimension weighted equally)
            loss = mse(output, target)
            total_loss += loss

            # Backward
            grad = mse_grad(output, target)
            model.backward(grad, lr)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

    # -----------------------------
    # Evaluation
    # -----------------------------
    print("\nFinal Evaluation:\n")

    for i, student in enumerate(students[:3]):
        x = student.reshape(1, -1)
        output = model.forward(x)

        print(f"Student {i+1}")
        print("Input:", x)
        print("Output:", output)
        print("-" * 40)
