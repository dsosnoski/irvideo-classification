
import tensorflow as tf


class ModelWithCertainty(tf.keras.models.Model):

    def __init__(self, threshold=.5, **params):
        super().__init__(**params)
        self._threshold = threshold

    def train_step(self, data):
        x, y_actual = data

        with tf.GradientTape() as tape:
            y_pred, y_certpred = self(x, training=True)  # Forward pass
            # 'actual' certainty should be 1 if we have the correct prediction as a 1, and should be 0 if we have the
            # correct prediction as a 0. It's not clear how to handle intermediate values, but for now use the
            # threshold to convert to 0/1
            y_actualfloat = tf.cast(y_actual, tf.float64)
            y_predfloat = tf.math.ceil(tf.cast(y_pred, tf.float64) - self._threshold)
            ycert = tf.math.reduce_sum(y_predfloat * y_actualfloat, axis=1)
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
        x, y = data
        y_pred, y_certpred = self(x, training=False)
        y_actualfloat = tf.cast(y, tf.float64)
        y_predfloat = tf.math.ceil(tf.cast(y_pred, tf.float64) - self._threshold)
        ycert = tf.math.reduce_sum(y_predfloat * y_actualfloat, axis=1)
        # Updates stateful loss metrics.
        self.compiled_loss([y, ycert], [y_pred, y_certpred], regularization_losses=self.losses)
        self.compiled_metrics.update_state([y, ycert], [y_pred, y_certpred])
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
