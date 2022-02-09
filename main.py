from network import NueralNetwork
import numpy as np

with np.load("mnist.npz") as data:
    training_images = data["training_images"]
    training_labels = data["training_labels"]
    print(len(data["training_images"][0]))

layer_sizes = (784,5,10)

net = NueralNetwork(layer_sizes)
net.print_accuracy(training_images, training_labels)