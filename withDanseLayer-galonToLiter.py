import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#                                                ---> We accept that our liquid is water <---

galon = np.array([3,8,11,19,27,39,52,18.7562,25.3605])
liter = np.array([11.3562, 30.2832, 41.6395, 71.9228, 102.2061, 147.6310, 196.8414, 71, 96])

for i,c in enumerate(galon):
    print("{} galon equal {} liter".format(c,liter[i]))

l0 = tf.keras.layers.Dense(units=4, input_shape=[1])  # units = number of layers   input_sahape = number of input
l1 = tf.keras.layers.Dense(units=4)
l2 = tf.keras.layers.Dense(units=1)  # we will have one output

model = tf.keras.Sequential([l0,l1,l2])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
training = model.fit(galon, liter, epochs=1000, verbose=2) # epochs : number of tarining , verbose must be 0,1,2 or True/False

print("Finished the training the model")

# Let's look at the epochs chart #
plt.xlabel("Epoch Number")
plt.ylabel("Loss Magnitude")
plt.plot(training.history['loss'])
plt.show()
###########################################
# Let's trying the model
answer = model.predict([83.23])
# liter = galon/0,26417 --- (83.23 galon = 315.0598)
print(answer)
