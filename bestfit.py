from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import tempfile
import subprocess
import automobile_data


log_dir = tempfile.mkdtemp()
print("tensorboard-dir", log_dir)

subprocess.Popen(['pkill', '-f', 'tensorboard'])
subprocess.Popen(['tensorboard', '--logdir', log_dir])


parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int, help='number of training steps')
parser.add_argument('--price_norm_factor', default=1000., type=float, help='price normalization factor')


def main(argv):
    """ Build, train and evaluate the model """
    args = parser.parse_args(argv[1:])

    (train_x, train_y), (test_x, test_y) = automobile_data.load_data()

    train_y /= args.price_norm_factor
    test_y /= args.price_norm_factor

    # build the training dataset
    training_input_fn = tf.estimator.inputs.pandas_input_fn(x=train_x, y=train_y, bach_size=64, shuffle=True, num_epochs=None)

    # build the validation data set
    eval_input_fn = tf.estimator.inputs.pandas_input_fn(x=test_x, y=test_y, batch_size=64, shuffle=False)

    # build the estimator with simple linear regression
    # model = tf.estimator.LinearRegressor(feature_columns=automobile_data.features_columns(), model_dir=log_dir)

    # build the estimator with DNN regression
    model = tf.estimator.DNNRegressor(hidden_units=[50,30,10], feature_columns=automobile_data.features_columns(), model_dir=log_dir)

    # train the model
    # by default, the estimator log output every 100 steps
    model.train(input_fn=training_input_fn, steps=args.train_steps)

    # evaluate how the model performs on data it has not yet seen
    eval_result = model.evaluate(input_fn=eval_input_fn)

    # the evaluation retirns a python dictionary
    # the "average loss" key holds the Mean Suqared Error MSE
    average_loss = eval_result["average_loss"]

    # convert MSE to Root Mean Suqare Error
    print("\n" + 80 * "*")
    print("\nRMS error for the test set: ${:.0f}".format(args.price_norm_factor * average_loss ** 0.5))

    # run the model in prediction mode

    df = test_x[:2]
    pre_input_fn = tf.estimator.inputs.pandas_input_fn(x=df, shuffle=False)
    predict_results = model.predict(input_fn=pre_input_fn)

    # print prediction result
    print("\nPrediction results:")
    for i, prediction in enumerate(predict_results):
        print(args.price_norm_factor * prediction['predictions'])
    print()


if __name__ == "__main__":
    # the estimator periodically generates "INFO" logs; make the se logs visible
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)



