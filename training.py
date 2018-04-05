import time
import os

import numpy as np
import tensorflow as tf
import pickle

import data_input
import similarity




def train_net(train_dir='./train', val_dir=None, max_steps=100000, batch_size=128, max_n_images=1000, n_retrieved=60):

    # Load data
    sim_dir = './sim'
    similarity.create_dataset(train_dir, sim_dir, max_n_images, k_retrieved=n_retrieved)
    data = data_input.read_data_sets(train_dir, val_dir, sim_dir, max_n_images)
    n_comp_im = data.train.training_sim[0].shape[0]

    # Check target f-score
    label_dict = data_input.get_label_dict(train_dir, val_dir)
    # target_f_score = check_score.check_f_score(data, label_dict)
    # print('Target f-score: %.3f' % target_f_score)

    # Prepare graph data
    with tf.name_scope('data'):
        x = tf.placeholder(tf.float32, [None, 2048], name="input")
        y_ = tf.placeholder(tf.float32, [None, n_comp_im], name="label")
        keep_prob = tf.placeholder(tf.float32, name="dropout_prob")

    # Add feature to summary
    tf.image_summary('input', tf.reshape(x, [-1, 64, 32, 1]), 10)

    # Compute output
    with tf.name_scope('fc'):
        x_drop = tf.nn.dropout(x, keep_prob)
        fc8W = tf.Variable(tf.truncated_normal([2048, n_comp_im], stddev=0.01), name="fc")
        fc8b = tf.Variable(tf.zeros([n_comp_im]), name="bias")
        y_output = tf.matmul(x_drop, fc8W) + fc8b
        prob = tf.nn.sigmoid(y_output)
        tf.histogram_summary("weights", fc8W)
        tf.histogram_summary("biases", fc8b)
        tf.histogram_summary("y", y_output)

    # Loss
    with tf.name_scope("xent") as scope:
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_output, y_))
        tf.scalar_summary("cross-entropy", loss)
    with tf.name_scope("train") as scope:
        train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)

    # Saver
    saver = tf.train.Saver(tf.all_variables())

    # Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)

    # Merge summaries
    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(os.path.join(train_dir, 'logs'), sess.graph)

    # Parameters
    print('########################################')
    print('Epochs: %d' % ((max_steps * batch_size) // max_n_images))
    print('Learning rate:', 1e-4)
    print('Batch size:', batch_size)
    print('Number of training images:', max_n_images)
    print('Number of retrieved images:', n_retrieved)
    print('########################################')

    # Training

    tf.initialize_all_variables().run()
    if val_dir != None:
        val_data = data.validation.next_batch(batch_size)

    start = time.time()

    for i in range(max_steps):

        batch = data.train.next_batch(batch_size)

        if i % 1000 == 0:

            true_labels = np.float32(batch[1])

            train_loss, train_prob, summary = sess.run([loss, prob, summary_op],
                                                       feed_dict={
                                                           x: batch[0],
                                                           y_: true_labels,
                                                           keep_prob: 1.0
                                                       })

            f_score_train = 0
            for b_i in range(batch_size):
                sim_images_index = np.argpartition(train_prob[b_i], -n_retrieved)[-n_retrieved:]
                sim_images_ids = list(data.train.ref_order_ids[sim_images_index])
                f_score_train += data_input.score(label_dict, target=batch[3][b_i], selection=sim_images_ids, n=50)
            f_score_train /= batch_size

            if val_dir != None:
                f_score_val = 0
                val_prob = prob.eval(feed_dict={
                    x: val_data[0],
                    y_: true_labels,
                    keep_prob: 1.0
                })
                for b_i in range(batch_size):
                    sim_images_index = np.argpartition(val_prob[b_i], -n_retrieved)[-n_retrieved:]
                    sim_images_ids = list(data.train.ref_order_ids[sim_images_index])
                    f_score_val += data_input.score(label_dict, target=val_data[3][b_i], selection=sim_images_ids, n=50)
                f_score_val /= batch_size

            end = time.time()

            if val_dir != None:
                print("[%d/%d] Training loss: %.3f || Scores: %.3f (train) / %.3f (val) (%.0f sec)"
                  % (i, max_steps, train_loss, f_score_train, f_score_val, (end - start)))
            else:
                print("[%d/%d] Training loss: %.3f || Scores: %.3f (train) (%.0f sec)"
                      % (i, max_steps, train_loss, f_score_train, (end - start)))

            start = time.time()

            summary_writer.add_summary(summary, i)

        train_op.run(feed_dict={x: batch[0], y_: true_labels, keep_prob: 0.5})

        if (i % 10000 == 0 or ((i + 1) == max_steps and i > 10000)) and i > 0:
            checkpoint_path = 'pretrained_model.ckpt'
            saver.save(sess, checkpoint_path, global_step=i)
            with open('ref_order.pickle', 'wb') as f:
                pickle.dump(data.train.ref_order_ids, f)

    # F-scores

    f_score_train = 0
    train_prob = prob.eval(feed_dict={
        x: data.train.images,
        y_: data.train.training_sim,
        keep_prob: 1.0
    })
    for b_i in range(data.train.images.shape[0]):
        sim_images_index = np.argpartition(train_prob[b_i], -n_retrieved)[-n_retrieved:]
        sim_images_ids = list(data.train.ref_order_ids[sim_images_index])
        f_score_train += data_input.score(label_dict, target=data.train.ids[b_i], selection=sim_images_ids, n=50)
    f_score_train /= data.train.images.shape[0]
    print('Training F-score: %.4f' % f_score_train)

    if val_dir != None:
        f_score_val = 0
        l = data.validation.images.shape[0]
        if data.train.images.shape[0] > data.validation.images.shape[0]:
            val_x = data.validation.images
            val_y = data.train.training_sim[0:l]
        else:
            val_x = data.validation.images[0:l]
            val_y = data.train.training_sim
        train_prob = prob.eval(feed_dict={
            x: val_x,
            y_: val_y,
            keep_prob: 1.0
        })
        for b_i in range(data.validation.images.shape[0]):
            sim_images_index = np.argpartition(train_prob[b_i], -n_retrieved)[-n_retrieved:]
            sim_images_ids = list(data.train.ref_order_ids[sim_images_index])
            f_score_val += data_input.score(label_dict, target=data.validation.ids[b_i], selection=sim_images_ids, n=50)
        f_score_val /= data.validation.images.shape[0]
        print('Validation F-score: %.4f' % f_score_val)
