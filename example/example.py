from adversarial_attack.adversarial_attack import AdversarialAttack
import keras

model_path = './simple_cnn_mnist.keras'
a = AdversarialAttack(model_path)

(x_train, y_train), (_, _) = keras.datasets.mnist.load_data()

a.attack(x_train[0], y_train[0], output_path='picture.png')