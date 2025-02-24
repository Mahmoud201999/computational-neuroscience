import numpy as np
input_size = 2
hidden_size = 2
output_size = 1
np.random.seed(42)
W1 = np.random.uniform(-0.5, 0.5, (input_size, hidden_size))
b1 = np.array([0.5, 0.5]) 
W2 = np.random.uniform(-0.5, 0.5, (hidden_size, output_size))
b2 = np.array([0.7])  
input_data = np.array([0.8, 0.3])
def tanh(x):
    return np.tanh(x)
def forward_propagation(input_data, W1, b1, W2, b2):
    hidden_input = np.dot(input_data, W1) + b1
    hidden_output = tanh(hidden_input)
    output_input = np.dot(hidden_output, W2) + b2
    output = tanh(output_input)
    return output
output = forward_propagation(input_data, W1, b1, W2, b2)
print("Network Output:", output)
print("\nW1:\n", W1)
print("\nb1:\n", b1)
print("\nW2:\n", W2)
print("\nb2:\n", b2)

