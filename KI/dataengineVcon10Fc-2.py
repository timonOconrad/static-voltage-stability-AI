import csv
import pandas as pd
import numpy as np
import time

def read_csv_file(file_path):
    data = []
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            data.append(row)
    return data
def data_load(file_path):

    data = read_csv_file(file_path)

    #print(data)
    df = pd.DataFrame(data)

    # Die Daten in separate Spalten aufteilen, indem wir das Trennzeichen ';' verwenden
    df = df[0].str.split(';', expand=True)

    # Die ersten 10 Spalten auswählen
    X = df.iloc[:, :10]

    # Die 11. Spalte auswählen
    y = df.iloc[:, 20]
    print(y)
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    return X, y

file_path_train = 'data/100power_new_train.csv'
#file_path_test = 'Data_for_dataengine/100power_test.csv'

file_path_test = 'data/100power_new_test.csv'
#file_path_test = 'Data_for_dataengine/1000new_test.csv'

X_train, y_train = data_load(file_path_train)
X_val, y_val = data_load(file_path_test)

# Ausgabe
print("Feature:")
print(X_train)

# 7 Feature


min_values = np.min(X_train, axis=0)
max_values = np.max(X_train, axis=0)

#min_values = np.array([ 0, 1, -88, -91, -78., 110, -10., -83,-63,-59])
#max_values = np.array([100, 178, -19, 0, -8,  163, 10, -12, 24, -19])

ranges = max_values - min_values

# Teile jeden Wert in der Matrix durch den entsprechenden Bereich und subtrahiere den minimalen Wert
normalized_data = (X_train - min_values) / ranges
normalized_testdata = (X_val - min_values) / ranges
#print(normalized_data)

print("\nFeature:")
print(X_train)
X_train =normalized_data
#X_train =np.delete(X_train, [0, 5, 6], axis=1)

X_val =normalized_testdata
#X_val =np.delete(X_val, [0, 5, 6], axis=1)

#X_val= X_train
#y_val = y_train

print("\nFeature:")
print(X_train)

print("\nTarget:")
print(y_train)

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from tensorflow import keras

# Modell initialisieren
model = Sequential()

# Erste verdeckte Schicht
model.add(Dense(3, input_dim=10, activation='sigmoid'))

# Zweite verdeckte Schicht
model.add(Dense(2, activation='sigmoid'))

# Ausgabeschicht
model.add(Dense(1, activation='linear'))

# Modell kompilieren

# Lernrate und Gewichtsverfall (Weight Decay) definieren
initial_learning_rate = 0.01
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,  # Wie oft soll die Lernrate aktualisiert werden
    decay_rate=0.99999999999,  # Gewichtsverfall
    staircase=True)

optimizer = keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.1)

metrics = [
    keras.metrics.FalseNegatives(name="fn"),
    keras.metrics.FalsePositives(name="fp"),
    keras.metrics.TrueNegatives(name="tn"),
    keras.metrics.TruePositives(name="tp"),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
]
start_time = time.time()
#model.compile(loss='mean_absolute_error', optimizer='adam')
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'] + metrics)

model.summary()
batch_size=50
epochs= 100000
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
end_time = time.time()

last_learning_rate = lr_schedule.get_config()['initial_learning_rate'] * (lr_schedule.get_config()['decay_rate'] ** 10000)
# Extrahieren Sie die aufgezeichneten Metriken aus dem `history`-Objekt.
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

import matplotlib.pyplot as plt



window_size =1000
# Erstellen Sie einen Plot für die Genauigkeit.
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(np.convolve(accuracy, np.ones(window_size)/window_size, mode='valid'), label='Training Accuracy')
plt.plot(np.convolve(val_accuracy, np.ones(window_size)/window_size, mode='valid'), label='Validation Accuracy')
plt.legend(loc='upper left')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of epochs')
plt.xscale('log')  # Skaliert die x-Achse logarithmisch
# Erstellen Sie einen Plot für den Verlust.
plt.subplot(1, 2, 2)
plt.plot(np.convolve(loss, np.ones(window_size)/window_size, mode='valid'), label='Training Loss')
plt.plot(np.convolve(val_loss, np.ones(window_size)/window_size, mode='valid'), label='Validation Loss')
plt.legend(loc='upper left')
plt.title('Training and Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Number of epochs')
plt.xscale('log')  # Skaliert die x-Achse logarithmisch
num_data_points = len(X_train)

duration = end_time - start_time
max_index = np.argmax(val_accuracy)
max_val_accuracy = val_accuracy[max_index]
value_at_max_accuracy = accuracy[max_index]
max_acc_index = np.argmax(accuracy)
value_at_max_val_accuracy = val_accuracy[max_acc_index]

text = f"PowerVcon10F, Variant c-2, Number of training samples: {num_data_points}, batch size: {batch_size}, learning rate: {np.round(last_learning_rate,6)}, max accuracy {np.round(max(accuracy),4)},val accuracy at max accuracy {np.round(value_at_max_val_accuracy,4)},  max var accuracy {np.round(max_val_accuracy,4)}, accuracy at max var accuracy {np.round(value_at_max_accuracy,4)}, Time taken {np.round(duration,4)} s"
plt.text(-1.5, -0.2, text, va='bottom', transform=plt.gca().transAxes, fontsize=6)
plt.savefig('Power10F_Variant_c2_accuracy.png', bbox_inches='tight')



plt.show()
# Trainingsmetriken
fn = history.history['fn']
fp = history.history['fp']
tn = history.history['tn']
tp = history.history['tp']

# Validierungsmetriken
val_fn = history.history['val_fn']
val_fp = history.history['val_fp']
val_tn = history.history['val_tn']
val_tp = history.history['val_tp']
# Plot für Trainingsdaten
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(np.convolve(fn, np.ones(window_size)/window_size, mode='valid'), label='False Negatives')
plt.plot(np.convolve(fp, np.ones(window_size)/window_size, mode='valid'), label='False Positives')
plt.plot(np.convolve(tn, np.ones(window_size)/window_size, mode='valid'), label='True Negatives')
plt.plot(np.convolve(tp, np.ones(window_size)/window_size, mode='valid'), label='True Positives')
plt.legend(loc='lower left')
plt.ylabel('Number of the assignment')
plt.xlabel('Number of epochs')
plt.title('Training Metrics per Epoch')
plt.xscale('log')  # Skaliert die x-Achse logarithmisch


# Plot für Validierungsdaten
plt.subplot(1, 2, 2)
plt.plot(np.convolve(val_fn, np.ones(window_size)/window_size, mode='valid'), label='False Negatives')
plt.plot(np.convolve(val_fp, np.ones(window_size)/window_size, mode='valid'), label='False Positives')
plt.plot(np.convolve(val_tn, np.ones(window_size)/window_size, mode='valid'), label='True Negatives')
plt.plot(np.convolve(val_tp, np.ones(window_size)/window_size, mode='valid'), label='True Positives')
plt.legend(loc='lower left')
plt.ylabel('Number of the assignment')
plt.xlabel('Number of epochs')
plt.title('Validation Metrics per Epoch')
plt.xscale('log')  # Skaliert die x-Achse logarithmisch

text = f"PowerVcon10F, Variant c-2, Number of training samples: {num_data_points}, batch size: {batch_size}, learning rate: {np.round(last_learning_rate,6)}, max accuracy {np.round(max(accuracy),4)},val accuracy at max accuracy {np.round(value_at_max_val_accuracy,4)},  max var accuracy {np.round(max_val_accuracy,4)}, accuracy at max var accuracy {np.round(value_at_max_accuracy,4)}, Time taken {np.round(duration,4)} s"
plt.text(-1.5, -0.2, text, va='bottom', transform=plt.gca().transAxes, fontsize=6)
plt.savefig('Power10F_Variant_c2_metric.png', bbox_inches='tight')

plt.show()
