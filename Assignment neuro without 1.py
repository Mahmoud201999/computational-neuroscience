def random_weight():
    return (0.5 - ((id(object()) % 1000) / 1000)) 
input_size = 2
hidden_size = 2
output_size = 1
W1 = [[random_weight() for _ in range(hidden_size)] for _ in range(input_size)]
b1 = [0.5, 0.5]
W2 = [[random_weight()] for _ in range(hidden_size)]
b2 = [0.7]
input_data = [0.8, 0.3]
def tanh(x):
    return (2 / (1 + pow(2.718, -2 * x))) - 1  
def forward_propagation(input_data, W1, b1, W2, b2):
    hidden_input = [sum(input_data[j] * W1[j][i] for j in range(input_size)) + b1[i] for i in range(hidden_size)]
    hidden_output = [tanh(h) for h in hidden_input]
    output_input = sum(hidden_output[i] * W2[i][0] for i in range(hidden_size)) + b2[0]
    output = tanh(output_input)
    return output
output = forward_propagation(input_data, W1, b1, W2, b2)
print("Output of the network:", output)
print("\nW1:\n", W1)
print("\nb1:\n", b1)
print("\nW2:\n", W2)
print("\nb2:\n", b2)

