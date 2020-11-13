import tensorflow as tf
@tf.function
def regression_loss(y_true, y_pred):
    dif_x = tf.math.square(y_true[:,0] - y_pred[:,0])
    dif_y = tf.math.square(y_true[:,1] - y_pred[:,1])
    dist_vector = tf.math.sqrt(dif_x + dif_y)
    loss = tf.math.reduce_mean(dist_vector)
    return loss