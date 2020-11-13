import tensorflow.keras as keras
from classification_models.keras import Classifiers

def base_model():
    Resnet18, _ = Classifiers.get('resnet18')
    input_image = keras.layers.Input(shape=(None,None,3))

    base_model = Resnet18(input_tensor=input_image, weights='imagenet', include_top=False)

    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(64, activation='relu', name='Middle_Dense_1')(x)
    x = keras.layers.Dense(32, activation='relu', name='Middle_Dense_2')(x)
    x = keras.layers.Dense(16, activation='relu', name='Middle_Dense_3')(x)
    regression = keras.layers.Dense(2, activation='linear', name='regression_layer')(x)

    model = keras.models.Model(inputs=[base_model.input], outputs=[regression])

    for layer in model.layers:
        layer.trainable = True
    model.summary()
    return model