
import tensorflow as tf


class ModelWithCertainty(tf.keras.models.Model):

    def __init__(self, threshold=.5, **params):
        super().__init__(**params)
        self._threshold = threshold

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y_actual = data

        with tf.GradientTape() as tape:
            y_pred, y_certpred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            # certainty should be 1 if we have the correct prediction as a 1, and should be 0 if we have the correct
            # prediction as a 0. So just use the product of y_pred with y_actual
            y_actualfloat = tf.cast(y_actual, tf.float32)
            ycert = tf.math.reduce_sum(y_pred * y_actualfloat, axis=1)
            loss = self.compiled_loss([y_actual, ycert], [y_pred, y_certpred], regularization_losses=self.losses)
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state([y_actual, ycert], [y_pred, y_certpred])
        # Return a dict mapping metric names to current value
        metrics = {m.name: m.result() for m in self.metrics}
        return metrics

    def test_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        y_pred, y_cert = self(x, training=False)
        yfloat = tf.cast(y, tf.float64)
        y_predfloat = tf.math.ceil(tf.cast(y_pred, tf.float64) - self._threshold)
        ycert = tf.math.reduce_sum(y_predfloat * yfloat, axis=1)
        # Updates stateful loss metrics.
        self.compiled_loss([y, ycert], [y_pred, y_cert], regularization_losses=self.losses)
        self.compiled_metrics.update_state([y, ycert], [y_pred, y_cert])
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
