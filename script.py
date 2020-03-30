import codecademylib3_seaborn
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

#Define posible Binary Inputs for a 2-Input Logic Gate
data = [[0, 0], [0, 1], [1, 0], [1, 1]]

#Define the XOR Logic Gate Output for the data defined before
labels = [0, 1, 1, 0]

#Creates a Scatter Plot to Graph the Decision Boundaries
plt.scatter([point[0] for point in data], [point[1] for point in data], c = labels)

#Show the graph
plt.show()

#Creates a Perceptron with a Maximun of 40 Iterations
classifier = Perceptron(max_iter = 40)

#Train the Classifier with the Data and Output
classifier.fit(data, labels)

#Creates two numerical array from 0 to 100 to Graph the Heatmap of the model
x_values = np.linspace(0, 1, 100)
y_values = np.linspace(0, 1, 100)

#Prepare the Value List to plot the Heatmap
point_grid = list(product(x_values, y_values))
distances = classifier.decision_function(point_grid)
abs_distances = [abs(x) for x in distances]
distances_matrix = np.reshape(abs_distances, (100, 100))

#Creates the Heatmap to plot
heatmap = plt.pcolormesh(x_values, y_values, distances_matrix)

#Adds the Heatmap to the graph
plt.colorbar(heatmap)

#Show the graph
plt.show()