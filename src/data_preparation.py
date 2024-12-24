import tensorflow as tf
import pickle
import os
import numpy as np


def load_data():
    """Carga el dataset CIFAR-10 desde la carpeta '../data'."""
    data_dir = "../data/cifar-10-batches-py"

    def load_batch(file):
        with open(file, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
        return batch[b'data'], batch[b'labels']

    x_train, y_train = [], []
    for i in range(1, 6):
        data, labels = load_batch(os.path.join(data_dir, f"data_batch_{i}"))
        x_train.append(data)
        y_train.extend(labels)
    
    """Normalizar datos"""
    x_train = np.vstack(x_train).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    x_train = x_train / 255.0

    data, labels = load_batch(os.path.join(data_dir, "test_batch"))
    x_test = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
    y_test = labels

    """Mantener etiquetas como enteros"""
    y_train, y_test = np.array(y_train).flatten(), np.array(y_test).flatten()

    return (x_train, y_train), (x_test, y_test)



def create_datasets(x_train, y_train, x_test, y_test, batch_size=32):
    """Crea datasets de entrenamiento, validación y prueba."""
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size).shuffle(10000)
    val_dataset = train_dataset.take(int(0.2 * len(train_dataset)))  # 20% para validación
    train_dataset = train_dataset.skip(int(0.2 * len(train_dataset)))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    return train_dataset, val_dataset, test_dataset