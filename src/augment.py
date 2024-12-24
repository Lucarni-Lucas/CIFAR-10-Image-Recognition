import tensorflow as tf

def augment_data(dataset):
    """Aplica aumento de datos al dataset."""
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
    ])
    
    dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32), y))
    
    augmented_dataset = dataset.map(lambda x, y: (data_augmentation(x), y))
    
    return dataset.concatenate(augmented_dataset)