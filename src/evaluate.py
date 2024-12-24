from tensorflow import keras
from data_preparation import load_data, create_datasets

(x_train, y_train), (x_test, y_test) = load_data()
_, _, test_dataset = create_datasets(x_train, y_train, x_test, y_test)

model = keras.models.load_model('../results/model_cifar10.keras')
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.2f}, Test loss: {test_loss:.2f}")