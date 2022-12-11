# from sklearn.neural_network import MLPClassifier
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score
# import numpy as np
# #from sklearn import neural_network

# data = []
# expectedOutput = []

# def importData(file):
#     with open(file, 'r') as f:
#         lines = f.readlines()
#         for i in lines:
#             currLine = i.split()
#             for j in range(len(currLine)):
#                 currLine[j] = float(currLine[j])
#                 if currLine[j] == float(0) or currLine[j] == float(1):
#                     expectedOutput.append(int(currLine[j]))
#                     currLine.remove(currLine[j])
#             data.append(currLine)
#     return len(data)

# importData("data3.txt")

# X_train, X_test, Y_train, Y_test = train_test_split(data, expectedOutput, stratify=expectedOutput, random_state=2, test_size=0.34)

# mlp = MLPClassifier(random_state=1, max_iter=300, hidden_layer_sizes=(10, 5), verbose=True, early_stopping=False, warm_start=True, solver="adam", validation_fraction=0.33).fit(X_train, Y_train)

# # epoch = mlp(verbose=1)
# # value = mlp.verbose=2

# # plt.plot(np.array(epoch))
# # plt.plot(np.array(value))

# # print(str(epoch))

# mlp.score(X_train, Y_train)

# plt.plot(mlp.loss_curve_)
# # plt.plot(mlp.validation_scores_)
# # print(mlp.score)

# plt.show()





# mlp.predict_proba(X_test)
# Y_prediction = mlp.predict(X_test)
# # Y_prediction = mlp.predict(Y_test)
# # mlp.score(Y_train, Y_test)
# # plt.plot(mlp.score(X_test, Y_test))

# print(mlp.score(X_test, Y_test))

# accuracy_test = accuracy_score(Y_test, Y_prediction)

# print(accuracy_test)

# print(mlp.loss_history)

# plt.plot(mlp.loss_curve_)
# plt.plot(mlp.loss_curve_)

# plt.show()



# mlp.fit(train, trainOutput)

# mlp.predict(train)

# mlp.score(trainOutput, testOutput)


# mlp.predict_proba(test)

# mlp = MLPClassifier(random_state=1, max_iter=300).fit(train)
# mlp.predict_proba(test)


# mlp = MLPClassifier(random_state=5)

#######################################################################################################################################

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import train_test_split

# X_train, X_test, Y_train, Y_test = train_test_split(data, expectedOutput, stratify=expectedOutput, random_state=2, test_size=0.34)

# mlp = MLPClassifier(hidden_layer_sizes=(50), max_iter=50, alpha=1e-4, solver='adam', verbose=0, tol=1e-8, random_state=1, learning_rate_init=.01).fit(X_train, Y_train)

# N_TRAIN_SAMPLES = X_train.shape[0]
# N_EPOCHS = 25
# N_BATCH = 128
# N_CLASSES = np.unique(Y_train)

# scores_train = []
# scores_test = []

# epoch = 0
# while epoch < N_EPOCHS:
#     print('epoch: ', epoch)
#     # while True:
#     scores_train.append(mlp.score(X_train, Y_train))

#     scores_test.append(mlp.score(X_test, Y_test))

#     epoch += 1

# fig, ax = plt.subplots(2)
# ax[0].plot(scores_train)
# ax[0].set_title('Train')
# ax[1].plot(scores_test)
# ax[1].set_title('Test')
# fig.suptitle("accuracy over epochs", fontsize=14)
# plt.show()


############################################################################################################################

import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt



data = []
expectedOutput = []

def importData(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        for i in lines:
            currLine = i.split()
            for j in range(len(currLine)):
                currLine[j] = float(currLine[j])
                if currLine[j] == float(0) or currLine[j] == float(1):
                    expectedOutput.append(int(currLine[j]))
                    currLine.remove(currLine[j])
            data.append(currLine)
    return len(data)

importData("data2.txt")

x = np.array(data)
y = np.array(expectedOutput)

model = Sequential()
model.add(Dense(8, input_shape = (8,), activation='sigmoid',))
model.add(Dense(6, activation='sigmoid'))
#model.add(Dense(4, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x, y, validation_split=0.33, epochs=50, batch_size=4)

returnData = model.evaluate(x, y, return_dict=True)

for k, v in returnData.items():
    print(k, v)

# metrics = model.get_metrics_result()

# for k, v in metrics.items():
#     print(k, v)


# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])

# plt.plot(model.density)
# plt.show()