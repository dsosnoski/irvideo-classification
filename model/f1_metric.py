
import tensorflow as tf


class F1ScoreMetric(tf.keras.metrics.Metric):
    """
    Adapted from code:
    https://stackoverflow.com/questions/35365007/tensorflow-precision-recall-f1-score-and-confusion-matrix
    'Multi-label case'
    """

    def __init__(self, threshold=0.5, **params):
        super().__init__(**params)
        self.score = 0
        self.threshold = threshold

    def update_state(self, y_true, y_pred, sample_weight=None):
        f1 = self.tf_f1_score(y_true, y_pred)
        self.score = f1
        return f1

    def result(self):
        return self.score

    def tf_f1_score(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float64)
        y_pred = tf.math.ceil(tf.cast(y_pred, tf.float64) - self.threshold)
        TP = tf.cast(tf.math.count_nonzero(y_pred * y_true), tf.float64)
        FP = tf.cast(tf.math.count_nonzero(y_pred * (y_true - 1)), tf.float64)
        FN = tf.cast(tf.math.count_nonzero((y_pred - 1) * y_true), tf.float64)

        precision = tf.math.divide_no_nan(TP, (TP + FP))
        recall = tf.math.divide_no_nan(TP, (TP + FN))
        f1 = 2 * precision * tf.math.divide_no_nan(recall, (precision + recall))
        return f1
