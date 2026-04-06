
---

# 🧠 `main.py`

```python
import numpy as np

# Activation function
def relu(x):
    return np.maximum(0, x)

# Simple layer
class Layer:
    def __init__(self, input_size, output_size, name):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.bias = np.zeros((1, output_size))
        self.name = name

    def forward(self, x):
        output = relu(np.dot(x, self.weights) + self.bias)
        print(f"{self.name} Output Shape: {output.shape}")
        return output


# Education Neural Network
class EducationNN:
    def __init__(self):
        self.layers = [
            Layer(5, 10, "Primary (Input Encoding)"),
            Layer(10, 20, "Middle School (Pattern Learning)"),
            Layer(20, 30, "High School (Abstraction)"),
            Layer(30, 25, "Undergraduate (Specialization)"),
            Layer(25, 15, "Masters (Deep Representation)"),
            Layer(15, 10, "PhD (Knowledge Generation)"),
            Layer(10, 5, "Postdoc (Expert Output)")
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x


# Simulated Input (Raw Student)
def generate_student():
    # 5 features: literacy, numeracy, curiosity, memory, attention
    return np.array([[0.2, 0.3, 0.5, 0.4, 0.6]])


if __name__ == "__main__":
    model = EducationNN()
    
    student_input = generate_student()
    print("Input (Raw Student):", student_input)

    output = model.forward(student_input)

    print("\nFinal Output (Expert Representation):", output)
