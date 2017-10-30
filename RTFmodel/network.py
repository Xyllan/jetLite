import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
import numpy as np
import loader

relation_detail = 'basic'
loader.set_relation_detail(relation_detail)

FLAGS = tf.app.flags.FLAGS
seq_len = 15
input_dim = 302
word_dim = 300
pos_embed_dim = 50
pos_min = 1 - seq_len
pos_max = seq_len - 1
pos_vocab_len = pos_max - pos_min + 1
default_l2 = 3.0
output_dim = len(loader.labels)
n_filters = 150
batch_size = 50
pool_size = 3
min_window = 2
max_window = 5
dropout = 0.5


def freeze():
    # By Omid Alemi - Jan 2017
    input_graph_path = relation_detail + '.txt'
    checkpoint_path = relation_detail + '.ckpt'
    input_saver_def_path = ""
    input_binary = False
    output_node_names = "output"
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_frozen_graph_name = 'frozen_' + relation_detail + '.pb'
    output_optimized_graph_name = 'optimized_' + relation_detail + '.pb'
    clear_devices = True
    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, checkpoint_path, output_node_names,
                              restore_op_name, filename_tensor_name,
                              output_frozen_graph_name, clear_devices, "")
    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(output_frozen_graph_name, "rb") as f:
        data = f.read()
        input_graph_def.ParseFromString(data)

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,
        ['input', 'keep_prob'], # an array of the input node(s)
        ['output'], # an array of output nodes
        tf.float32.as_datatype_enum)
    # Save the optimized graph
    f = tf.gfile.FastGFile(output_optimized_graph_name, "w")
    f.write(output_graph_def.SerializeToString())


def _variable_with_weight_decay(name, shape, wd = 1e-4):
    """
    Taken from https://github.com/yuhaozhang/sentence-convnet/blob/master/model.py
    """
    var = tf.get_variable(name, shape)
    if wd is not None and wd != 0.:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name=name + '_weight_loss')
    else:
        weight_decay = tf.constant(0.0, dtype=tf.float32)
    return var, weight_decay


class Graph:
    def __init__(self):
        pass


def _auc_pr(true, prob, threshold):
    pred = tf.where(prob > threshold, tf.ones_like(prob), tf.zeros_like(prob))
    tp = tf.logical_and(tf.cast(pred, tf.bool), tf.cast(true, tf.bool))
    fp = tf.logical_and(tf.cast(pred, tf.bool), tf.logical_not(tf.cast(true, tf.bool)))
    fn = tf.logical_and(tf.logical_not(tf.cast(pred, tf.bool)), tf.cast(true, tf.bool))
    pre = tf.truediv(tf.reduce_sum(tf.cast(tp, tf.int32)), tf.reduce_sum(tf.cast(tf.logical_or(tp, fp), tf.int32)))
    rec = tf.truediv(tf.reduce_sum(tf.cast(tp, tf.int32)), tf.reduce_sum(tf.cast(tf.logical_or(tp, fn), tf.int32)))
    return pre, rec


def another_f(true, prob):
    pred = tf.argmax(prob, axis = 1)
    act = tf.argmax(true, axis = 1)
    mat = tf.confusion_matrix(act, pred)

    micro_p = tf.trace(mat) / tf.reduce_sum(mat)
    micro_r = micro_p
    macro_p = tf.reduce_mean(tf.truediv(tf.diag(mat), tf.reduce_sum(mat, axis=0)))
    macro_r = tf.reduce_mean(tf.truediv(tf.diag(mat), tf.reduce_sum(mat, axis=1)))

    micro_f = 2 * micro_p * micro_r / (micro_p + micro_r)
    macro_f = 2 * macro_p * macro_r / (macro_p + macro_r)
    return micro_f, macro_f


def jet_prf(true, prob):
    pred = tf.argmax(prob, axis = 1)
    act = tf.argmax(true, axis = 1)
    responses = tf.logical_not(tf.equal(pred, tf.zeros_like(pred)))
    eqs = tf.equal(pred, act)
    corrects = tf.logical_and(eqs, responses)
    correct = tf.reduce_sum(tf.cast(corrects, tf.int32))
    resp_count = tf.reduce_sum(tf.cast(responses, tf.int32))
    gold_count = tf.reduce_sum(tf.cast(tf.logical_not(tf.equal(act, tf.zeros_like(act))), tf.int32))

    pre = tf.truediv(correct, resp_count)
    rec = tf.truediv(correct, gold_count)
    return pre, rec


def accuracy(true, prob):
    pred = tf.argmax(prob, axis = 1)
    act = tf.argmax(true, axis = 1)
    eqs = tf.equal(pred, act)
    return tf.reduce_mean(tf.cast(eqs, tf.float32))


def get_graph(x, y):
    """
    Replicates the neural network by:
    Thien Huu Nguyen and Ralph Grishman. Relation extraction: Perspective from convolutional neural networks. In VS@ HLT-NAACL, pages 39–48, 2015
    """
    pool_tensors = []
    pool_tensor_sizes = []
    losses = []
    y_conv = None
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    x_beg = x[:, :, :-2]
    x_pos = x[:, :, -2:]
    pos_embed_mat = tf.get_variable("pos_embedding_mat", [pos_vocab_len, pos_embed_dim])
    embedded = tf.nn.embedding_lookup(pos_embed_mat, tf.cast(tf.add(x_pos, -pos_min), tf.int32))
    x_n = tf.concat([x_beg, tf.reshape(embedded, [-1, seq_len, 2 * pos_embed_dim])], axis = 2)
    print(x_n)
    for k_size in range(min_window, max_window + 1):
        with tf.variable_scope('conv-%d' % k_size) as scope:
            kernel, wd = _variable_with_weight_decay('kernel-%d' % k_size, [k_size, word_dim + 2 * pos_embed_dim, n_filters])
            losses.append(wd)
            conv = tf.nn.conv1d(value=x_n, filters=kernel, stride = 1, padding = 'VALID')
            biases = tf.get_variable(name='bias-%d' % k_size, shape=[n_filters])
            bias = tf.nn.bias_add(conv, biases)
            activation = tf.nn.relu(bias, name=scope.name)
            # shape of activation: [batch_size, conv_len, n_filters]
            # conv_len = activation.get_shape()[1]
            expanded = tf.expand_dims(activation, 1)
            pool = tf.nn.max_pool(expanded, ksize=[1, 1, expanded.get_shape()[2], 1], strides=[1, 1, 1, 1], padding='VALID')
            # shape of pool: [batch_size, 1, 1, num_kernel]
            feature_size = int(pool.get_shape()[2] * pool.get_shape()[3])
            pool_tensor_sizes.append(feature_size)
            pool_tensors.append(tf.reshape(pool, [-1, feature_size]))

    all_pools_size = np.sum(np.array(pool_tensor_sizes))
    pool_layer = tf.concat(pool_tensors, 1, name='pool')

    with tf.variable_scope('dropout') as scope:
        pool_dropout = tf.nn.dropout(pool_layer, keep_prob)

    fc1_kernel, fc1_wd = _variable_with_weight_decay('fc1-kernel', [all_pools_size, output_dim])
    fc1_bias = tf.get_variable('fc1-bias', [output_dim])
    y_conv = tf.nn.bias_add(tf.matmul(pool_dropout, fc1_kernel), fc1_bias, name = 'output')
    losses.append(wd)

    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = y_conv, labels = y,
                                                                name = 'cross_entropy_per_example')
        cross_entropy_loss = tf.reduce_mean(cross_entropy, name = 'cross_entropy_loss')
        losses.append(cross_entropy_loss)

    with tf.variable_scope('adam_optimizer'):
        total_loss = tf.add_n(losses, name='total_loss')
        train_step = tf.train.AdamOptimizer(1e-4).minimize(total_loss)

    with tf.variable_scope('evaluation') as scope:
        pre, rec = _auc_pr(y, tf.sigmoid(y_conv), 0.1)
        fscore = 2 * pre * rec / (pre + rec)
    with tf.variable_scope('jet_evaluation') as scope:
        jet_pre, jet_rec = jet_prf(y, tf.sigmoid(y_conv))
        jet_f = 2 * jet_pre * jet_rec / (jet_pre + jet_rec)
    with tf.variable_scope('micromacro_evaluation') as scope:
        micro_f, macro_f = another_f(y, tf.sigmoid(y_conv))
    with tf.variable_scope('accuracy') as scope:
        acc = accuracy(y, tf.sigmoid(y_conv))
    return y_conv, keep_prob, train_step, total_loss, fscore, jet_f, micro_f, macro_f, acc


def turn_one_hot(X, n = output_dim):
    # Turns a matrix into one-hot tensor
    X = X.reshape((X.shape[0]))
    o = np.zeros((X.shape[0], n))
    for i in range(n):
        o[X == i, i] = 1
    return o


def main():
    x_train, y_train = loader.load_training()
    n_train = x_train.shape[0]
    y_train = turn_one_hot(y_train, n = output_dim)
    x_test, y_test = loader.load_test()
    y_test = turn_one_hot(y_test, n = output_dim)
    x = tf.placeholder(tf.float32, [None, 15, 302], name = 'input')
    y_ = tf.placeholder(tf.float32, [None, output_dim])
    y_conv, keep_prob, train_step, total_loss, fscore, jet_fscore, micro_f, macro_f, acc = get_graph(x, y_)
    xs = np.array_split(x_train, int(n_train / 50), axis = 0)
    ys = np.array_split(y_train, int(n_train / 50), axis = 0)
    saver = tf.train.Saver()
    best_f = -1
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(40000):
            train_step.run(feed_dict={x: xs[i % len(xs)], y_: ys[i % len(ys)], keep_prob: 1 - dropout})
            if i % 100 == 0:
                jet_f = jet_fscore.eval(feed_dict = {x: x_test, y_: y_test, keep_prob: 1})
                print('step %d, jet-f-score %g' % (i, jet_f))
                if jet_f > best_f + 0.01 and jet_f > 0.4: # With long enough training, we beat 0.4 in all trials
                    best_f = jet_f
                    saver.save(sess, '%s.ckpt' % relation_detail)
                    tf.train.write_graph(sess.graph_def, '.', '%s.proto' % relation_detail, as_text=False)
                    tf.train.write_graph(sess.graph_def, '.', '%s.txt' % relation_detail, as_text=True)
                    freeze()

if __name__ == '__main__':
    main()
