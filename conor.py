import pandas as pd
from keras.layers import Dense
from keras.models import Sequential

# load dataset
dataframe = pd.read_csv("Data2.csv", header=None)
dataset = dataframe.values

# split into input (X) and output (Y) variables
X = dataset[:,0:8].astype(float)
Y = dataset[:,8]

model = Sequential()
model.add(Dense(8, input_shape=(8,), activation='sigmoid',))
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(4, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, validation_split=0.33, epochs=300, batch_size=20)