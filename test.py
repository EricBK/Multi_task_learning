
import tensorflow as tf

file_name = "./file"

with tf.Session() as sess:
    var = tf.Variable(42, name='my_var')
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.save(sess,file_name)
    saver.export_meta_graph(file_name + '.meta')

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(file_name + '.meta')
    saver.restore(sess, file_name)
    print(sess.run(var))

    # new code that fails:
    saver = tf.train.Saver()
    saver.save(sess,file_name)
