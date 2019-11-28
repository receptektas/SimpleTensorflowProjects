import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

miles = np.array([1, 5, 7, 12, 14, 18, 20, 36, 49, 63, 70, 99, 124.274238] , dtype=float)
km = np.array([1.609344, 8.04672, 11.265408, 19.312128, 22.530816, 28.968192, 32.18688, 57.936384, 78.857856, 101.388672, 112.65408, 159.325056, 200] , dtype=float)

for i,c in enumerate(miles):
    print("{} miles equal {} kilometers".format(c,km[i]))

l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([l0])
model.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(0.1))

training = model.fit(miles, km, epochs=500, verbose=2)

print("** Finished the training **")

# GRAPH OF EPOCHS
plt.xlabel("Epochs Number")
plt.ylabel("Loss Magnitude")
plt.plot(training.history['loss'])
plt.show()

# miles = km * 0.621371
# km = miles × 1.609344

result = model.predict([52])
# 52 mi = 83.685888 km

print(result)
print("These are the layer variables : {}".format(l0.get_weights()))