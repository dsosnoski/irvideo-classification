
import os
import sys

import tensorflow as tf

from model.f1_metric import F1ScoreMetric


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    argv = sys.argv
    model_path = argv[1]
    output_path = argv[2]
    directory_path, _ = os.path.split(model_path)
    model = tf.keras.models.load_model(model_path, custom_objects={'f1score': F1ScoreMetric}, compile=False)
    model.save_weights(output_path)


if __name__ == '__main__':
    sys.exit(main())
