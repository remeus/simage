import numpy as np
import tensorflow as tf
import pickle




def infere(image_features, n_comp_im=1000, n_retrieved=60):


    with open('ref_order.pickle', 'rb') as f:
        ref_order = pickle.load(f)

    # Prepare graph data
    with tf.name_scope('data'):
        x = tf.placeholder(tf.float32, [1, 2048], name="input")

    # Compute output
    with tf.name_scope('fc'):
        fc8W = tf.Variable(tf.truncated_normal([2048, n_comp_im], stddev=0.01), name="fc")
        fc8b = tf.Variable(tf.zeros([n_comp_im]), name="bias")
        y_output = tf.matmul(x, fc8W) + fc8b  ###
        prob = tf.nn.sigmoid(y_output)
        tf.histogram_summary("weights", fc8W)
        tf.histogram_summary("biases", fc8b)
        tf.histogram_summary("y", y_output)

    # Saver
    saver = tf.train.Saver(tf.all_variables())

    # Session
    sess = tf.InteractiveSession()

    # Inference
    tf.initialize_all_variables().run()

    try:
        saver.restore(sess, 'pretrained_model.ckpt')
    except:
        pass

    x_prob = sess.run(prob, feed_dict={x: np.expand_dims(image_features, 0)})

    sim_images_index = np.argpartition(x_prob[0], -n_retrieved)[-n_retrieved:]
    sim_images_ids = list(ref_order[sim_images_index])

    return sim_images_ids





