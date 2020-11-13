# Synthetic-gaze-data-augmentation-for-improved-user-calibration
Model weights and network implementation for reproducibility of the paper "Synthetic gaze data augmentation for improved user calibration"


## Modules
We used Tensorflow as framework to build the architecture. Specifically, we used tensorflow.keras. 
An additional module necessary to build the model is the classification_models module, from which we used the pretrained Resnet-18 over imagenet. To install this module, please refer to:
https://github.com/qubvel/classification_models

## Pretrained model over U2Eyes
The pretrained model needs a custom object in order to be loaded by keras.
```
model = keras.models.load_model('path_to_file_h5', custom_objects={'regression_loss':regression_loss}
```

The regression_loss function is defined as:
```python
import tensorflow as tf
@tf.function
def regression_loss(y_true, y_pred):
    dif_x = tf.math.square(y_true[:,0] - y_pred[:,0])
    dif_y = tf.math.square(y_true[:,1] - y_pred[:,1])
    dist_vector = tf.math.sqrt(dif_x + dif_y)
    loss = tf.math.reduce_mean(dist_vector)
    return loss
```
