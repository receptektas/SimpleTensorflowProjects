import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


celsius = np.array([-40, -10,  0,  8, 15, 22,  38] , dtype=float)
fahrenheit = np.array([-40,  14, 32, 46, 59, 72, 100] , dtype=float)

for i,c in enumerate(celsius): # enumerate i : index degerini tutuyor , c = sırayla değerleri alıyor.
    print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit[i]))

l0 = tf.keras.layers.Dense(units=1, input_shape=[1]) # units = katman sayısı   input_sahape = girdi sayısı
model = tf.keras.Sequential([l0])  # model için bir session actık

model.compile(loss='mean_squared_error',  # loss kayıp fonk. kaybımızı hesaplamanın yolu hazır bır tane kullandık
              optimizer=tf.keras.optimizers.Adam(0.1))  # optimizer : ksybı azaltmak için değerleri ayarlamanın yolu yine hazır kullandık
                                                        # 0.1 atılan adım buyuklugu
                                                        # Sayısal Analiz kullanan Gradient Descent adlı bir optimizasyon işlemi ile gerçekleştirilir.

history = model.fit(celsius, fahrenheit, epochs=500, verbose=2)  # .fit bir tur egitim yontemi
                                                                     #  input , output , kac kez calıscak(epochs) , verbose : eğitim cubugu acık/kapalı(0,1,2)
print("Finished training the model")

#  epoch grafigine bakıyoz
plt.xlabel('Epoch Number')  # label name of x axis
plt.ylabel("Loss Magnitude") # label name of y axis
plt.plot(history.history['loss']) # çizilecek grafik
plt.show() # grafiği göster demek

#tahmin etmesi için deger veriyoruz
print(model.predict([100.0]))

# fahrenhait = (celsius * 1.8) + 32

print("These are the layer variables: {}".format(l0.get_weights()))