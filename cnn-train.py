import argparse
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
from sklearn import preprocessing

class ConsistenLabelBinarizer(preprocessing.LabelBinarizer):
    """
    Create a more consistent version of sklearn's LabelBinarizer

    LabelBinarizer returns different values for binary and multiclass cases
    See http://stackoverflow.com/questions/31947140/sklearn-labelbinarizer-returns-vector-when-there-are-2-classes
    Hence, if the labels are in [0,1], the LabelBinarizer will only 
    return a single element array for transform, e.g. 0 -> [0] or 1-> [1]
    However, if the labels are in [0,1,2], it will return a 3-din array
    e.g. 0 -> [1 0 0], 2 -> [0 0 1]

    ConsistentLabelBinarizer fixes the behavior of the binary case to match
    multi-class
    """
    def transform(self, y):
        Y = super(ConsistenLabelBinarizer, self).transform(y)
        if self.y_type_ == 'binary':
            return np.hstack((Y, 1-Y))
        else:
            return Y

    def inverse_transform(self, Y, threshold=None):
        if self.y_type_ == 'binary':
            return super(ConsistenLabelBinarizer, self).inverse_transform(Y[:, 0], threshold)
        else:
            return super(ConsistenLabelBinarizer, self).inverse_transform(Y, threshold)

def load_data(filepath, delimiter, label_classes=None):
    """
    Load the training/test data from the provided file
    
    Input schema expected: Label <args.delimiter> Word_Id_Vector

    - Converts the Label into a one-vs-all numpy array
    - Converts the Word_Id_Vector into numpy array
    """
    labels = []
    vectors = []

    print("Reading data from {}".format(filepath))
    with open(filepath, 'r') as f_train:
        for line in f_train:
            cols = line.split(delimiter)

            label_str = cols[0]
            labels.append(label_str)

            vector_str = cols[1]
            vector = map(int, vector_str.split(" "))
            vectors.append(vector)
    print("Read {} lines".format(len(labels)))

    # Get the label classes
    lb = ConsistenLabelBinarizer()
    if label_classes is None:
        lb.fit(labels)
        label_classes = lb.classes_
    else:
        lb.fit(label_classes)

    print("Label classes: {}".format(label_classes))

    # Transform the multi-class labels into one-vs-all numpy array
    labels = lb.transform(labels)

    # Transform the vectors into numpy arary
    vectors = np.asarray(vectors)

    return labels, vectors, label_classes

def main():
    """
    Train a sentence classification CNN model
    """

    # Parse command line args
    # ==================================================
    parser = argparse.ArgumentParser(description='Train CNN model for text classification')

    parser.add_argument('-tr', '--train', required=True,
        help='Path to training data')
    parser.add_argument('-ev', '--eval', required=True,
        help='Path to evaluation data')
    parser.add_argument('-vs', '--vocab_size', required=True,
        help='Path to file containing vocab size')
    parser.add_argument('-d', '--delimiter', required=True, default='\t', 
        help='Column delimiter between row and label')
    parser.add_argument('-o', '--output_dir', required=True,
        help='Path to output checkpoints dir')
    parser.add_argument('-os', '--summary_dir', required=True,
        help='Path to output summaries dir')

    args, unknown = parser.parse_known_args()
    # Unescape the delimiter
    args.delimiter = args.delimiter.decode('string_escape')

    # Convert args to dict
    vargs = vars(args)

    print("\nArguments:")
    for arg in vargs:
        print("{}={}".format(arg, getattr(args, arg)))

    # Parameters
    # ==================================================

    # Model Hyperparameters
    tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
    tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
    tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
    tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
    tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

    # Training parameters
    tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
    tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
    tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
    tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    # Read and load input data
    # ==================================================
    print("Processing training data")
    y_train, x_train, label_classes = load_data(args.train, args.delimiter, label_classes=None)
    print("Processing evaluation data")
    y_dev, x_dev, dummy = load_data(args.eval, args.delimiter, label_classes=label_classes)

    # Get the vocab size
    with open(args.vocab_size, 'r') as f:
        vocab_size = int(f.readline())

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=vocab_size,
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.merge_summary(grad_summaries)

            # Summaries for loss and accuracy
            loss_summary = tf.scalar_summary("loss", cnn.loss)
            acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(args.summary_dir, "train")
            train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(args.summary_dir, "dev")
            dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = args.output_dir
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.all_variables())

            # Initialize all variables
            sess.run(tf.initialize_all_variables())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

if __name__ == '__main__':
    main()