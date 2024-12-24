from data_preparation import load_data, create_datasets
from augment import augment_data
from model import create_model
import matplotlib.pyplot as plt
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = load_data()
train_dataset, val_dataset, _ = create_datasets(x_train, y_train, x_test, y_test)

# Aumentar datos
train_dataset = augment_data(train_dataset)

# Detención de entrenamiento ante sobreajuste
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',        
    patience=3,                
    restore_best_weights=True  
)

# Crear y entrenar modelo
model = create_model()

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=15,
    callbacks=[early_stopping]
)

model.save('../results/model_cifar10.keras')

# Graficar métricas
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.legend()
plt.savefig('../results/training_curves_accuracy.png')
plt.close()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.savefig('../results/training_curves_loss.png')