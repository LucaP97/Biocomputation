import matplotlib.pyplot as plt
import sklearn.datasets as skl_data
import sklearn.neural_network as skl_nn

data, labels = skl_data.fetch_openml('mnist_784', version=1, return_X_y=True)
data = data / 255.0

mlp = skl_nn.MLPClassifier(hidden_layer_sizes=(50,), max_iter = 50, verbose=1, random_state=1)

data_train = data[0:63000]
labels_train = labels[0:63000]

data_test = data[63001:]
labels_test = labels[63001:]

mlp.fit(data_train, labels_train)
print("training set score", mlp.score(data_train, labels_train))
print("testing set score", mlp.score(data_test, labels_test))