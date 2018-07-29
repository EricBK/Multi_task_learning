import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.ops import array_ops
import pandas as pd
import os
import datetime
import logging
from MTL.data_util import get_data_batch,parser
args = parser.parse_args()
tf.app.flags.DEFINE_integer('input_dim',args.input_dim,'input dim')
tf.app.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability')
tf.app.flags.DEFINE_integer('num_epochs', 1, 'Number of epochs for training')
tf.app.flags.DEFINE_integer('num_classes', 2, 'Number of classes')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size')
tf.app.flags.DEFINE_integer('Iteration',10000,'iterations')
tf.app.flags.DEFINE_string('log_dir','./logs','training log dir')
tf.app.flags.DEFINE_string('log_dir_newTask','./logs_newTask','new task log dir')
tf.app.flags.DEFINE_string('checkpoints_dir','./checkpoints/model','checkpoints dir')
tf.app.flags.DEFINE_string('checkpoints_dir_newTask','./checkpoints_newTask/model','new task checkpoints dir')
tf.app.flags.DEFINE_string('train_mode','respective','choose training mode? respective or join')
tf.app.flags.DEFINE_string('mode','train','choose execute mode, train, test or finetune')
tf.app.flags.DEFINE_string('results_dir','./results','prediction results or save prediction pics')
#tf.app.flags.DEFINE_string('train_mode','join','choose training mode? respective or join')
TASK_LIST = [0,1]

tf.app.flags.DEFINE_integer('num_tasks', len(TASK_LIST), 'Number of tasks')
#tf.app.flags.DEFINE_integer('task_list', [0,1,2], 'list of tasks')
FLAGS = tf.app.flags.FLAGS

if not os.path.exists(FLAGS.results_dir):
    os.makedirs(FLAGS.results_dir)
def plot(losses=[],acc=[]):
    """
    画一个双坐标轴的图，左边y表示loss的变化，右边的y表示acc的变化
    :param losses:
    :param acc:
    :return:
    """
    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    for i in range(len(losses)):
        ax1.plot(losses[i])
    ax1.set_ylabel('Global loss')
    ax1.set_title('Loss and F1')
    ax1.set_xlabel('Iteration')
    ax1.legend(["task{} loss".format(i) for i in range(len(losses))],loc=4)
    colors = ['g-','r-','b-']
    color = [i for i in colors[:FLAGS.num_tasks]]
    if acc != []:
        ax2 = ax1.twinx()
        for i in range(len(acc)):
            ax2.plot(acc[i],color[i])
        ax2.set_ylabel('f1')
        ax2.legend(["task{} f1".format(i+1) for i in range(len(acc))],loc=7)
    plt.show()
def focal_loss(onehot_labels,cls_preds,alpha=0.25,gamma=2.0,name=None,scope=None,task_no=0):
    """

    :param onehot_labels:
    :param predictions:
    :param alpha:
    :param gamma:
    :param name:
    :param scope:
    :return: A 1-D tensor of length batchsize
    """
    with tf.name_scope(scope,'task_{}_focal_loss'.format(task_no),[cls_preds,onehot_labels]) as sc:
        logits = tf.convert_to_tensor(cls_preds)
        onehot_labels = tf.convert_to_tensor(onehot_labels)
        precise_logits = tf.cast(logits,tf.float32) if logits.dtype == tf.float16 else logits
        onehot_labels = tf.cast(onehot_labels,precise_logits.dtype)

        predictions = tf.nn.sigmoid(precise_logits)
        predictions_pt = tf.where(tf.equal(onehot_labels,1),predictions,1.-predictions)

        epsilon = 1e-8
        alpha_t = tf.scalar_mul(alpha, tf.ones_like(onehot_labels,dtype=tf.float32))
        alpha_t = tf.where(tf.equal(onehot_labels,1.0),alpha_t,1-alpha_t)
        losses = tf.reduce_sum(-alpha_t * tf.pow(1. - predictions_pt,gamma)*tf.log(predictions_pt+epsilon),
                               name=name,axis=1)
        return losses
class MTL(object):
    def __init__(self,mode):
        """

        :param task_num:
        :param xs:  placeholder for each task
        :param ys:  placeholder for each task
        """
        self.mode = mode
        if self.mode == 'train' or self.mode == 'finetune':
            self.keep_prob = 0.6
        elif self.mode == 'test':
            self.keep_prob = 1
        self._task_num = FLAGS.num_tasks

        self.task_outs = []     # 记录每个task 的最终输出
        self.task_outs_prob = []# 记录每个task 的最终输出的概率
        self.losses = []        # 记录每个task 的loss
        self.accuracies = []    # 计算每个task 的准确率
        self.f1_scores = []
        self.train_ops = []      # 每个task的优化

        # self.task_out     新任务的识别层输出
        # self.task_out_prob 新任务识别层输出概率
        # self.loss         新任务的loss
        # self.accuracy     新任务的 acc
        # self.train_op        新任务的 优化
    def add_placeholders(self):
        self.xs = [self.add_placeholder_x(name=i) for i in range(FLAGS.num_tasks)]
        self.ys = [self.add_placeholder_y(name=i) for i in range(FLAGS.num_tasks)]
    def add_placeholder_x(self,name='new'):
        #self.x = tf.placeholder(dtype=tf.float32,shape=[None,FLAGS.input_dim,1],name='inputs_{}'.format(name))
        self.x = tf.placeholder(dtype=tf.float32,shape=[None,FLAGS.input_dim],name='inputs_{}'.format(name))

        return self.x
    def add_placeholder_y(self,name='new'):
        self.y = tf.placeholder(dtype=tf.int32,shape=[None,FLAGS.num_classes],name='label_{}'.format(name))
        return self.y
    def inference(self,):
        # shared weights
        with tf.variable_scope('shared_weights') as shared_weights_scope:
            with tf.variable_scope('fc1') as fc1_scope:
                # onehot 类型的输入向量在输入时候需要连接一个全连接层
                if FLAGS.mode == 'train' or FLAGS.mode == 'test':
                    shape = int(np.prod(self.xs[0].get_shape()[1:]))
                elif FLAGS.mode == 'finetune':
                    shape = int(np.prod(self.x.get_shape()[1:]))
                weights = tf.get_variable('weights',initializer=tf.truncated_normal(shape=[shape,512],dtype=tf.float32,stddev=0.01))
                bias = tf.get_variable('bias',initializer=tf.constant(0.01,shape=[512],dtype=tf.float32))
                if FLAGS.mode == 'train' or FLAGS.mode == 'test':
                    self.shared_fc1 = []
                    for i in range(self._task_num):
                        tmp_out = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.xs[i],weights),bias))
                        tmp_out_dp = tf.nn.dropout(tmp_out,keep_prob=self.keep_prob)
                        self.shared_fc1.append(tmp_out_dp)
                elif FLAGS.mode == 'finetune':
                    self.shared_fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.x,weights),bias))

                tf.summary.histogram(fc1_scope.name+"weights",weights)
                tf.summary.histogram(fc1_scope.name+"bias",bias)

            with tf.variable_scope('fc2') as fc2_scope:
                weights = tf.get_variable('weights',
                                          initializer=tf.truncated_normal(shape=[512, 128], dtype=tf.float32,
                                                                          stddev=0.01))
                bias = tf.get_variable('bias', initializer=tf.constant(0.01, shape=[128], dtype=tf.float32))
                if FLAGS.mode == 'train' or FLAGS.mode == 'test':
                    self.shared_fc2 = []
                    for i in range(self._task_num):
                        tmp_out = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.shared_fc1[i], weights), bias))
                        tmp_out_dp = tf.nn.dropout(x=tmp_out,keep_prob=self.keep_prob)
                        self.shared_fc2.append(tmp_out_dp)
                elif FLAGS.mode == 'finetune':
                    self.shared_fc2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.shared_fc1,weights),bias))
                tf.summary.histogram(fc2_scope.name+"weights",weights)
                tf.summary.histogram(fc2_scope.name+"bias",bias)
            with tf.variable_scope('reshape') as reshape_scope:
                if FLAGS.mode == 'train' or FLAGS.mode == 'test':
                    for i in range(self._task_num):
                        self.shared_fc2[i] = tf.reshape(self.shared_fc2[i],shape=[-1, 128, 1])
                elif FLAGS.mode == 'finetune':
                    self.shared_fc2 = tf.reshape(self.shared_fc2,shape=[-1,128,1])
            with tf.variable_scope('conv1d_1') as conv1d_scope:
                self.shared_conv1d_1 = []
                self.shared_out_1 = []
                weights = tf.get_variable('weights',initializer=tf.truncated_normal(shape=[3,1,8],dtype=tf.float32,stddev=0.01))
                bias = tf.get_variable('bias',initializer=tf.constant(0.0,dtype=tf.float32,shape=[8]))
                if FLAGS.mode == 'train' or FLAGS.mode == 'test':
                    for i in range(self._task_num):
                        conv1d_tmp = tf.nn.conv1d(self.shared_fc2[i], weights, stride=1, padding='SAME')
                        out_tmp = tf.nn.relu(tf.nn.bias_add(conv1d_tmp,bias))
                        self.shared_conv1d_1.append(conv1d_tmp)
                        self.shared_out_1.append(out_tmp)
                elif FLAGS.mode == 'finetune':
                    conv1d_tmp = tf.nn.conv1d(self.shared_fc2, weights, stride=1, padding='SAME')
                    self.out_tmp = tf.nn.relu(tf.nn.bias_add(conv1d_tmp, bias))
                tf.summary.histogram(conv1d_scope.name+"weights",weights)
                tf.summary.histogram(conv1d_scope.name+"bias",bias)

            with tf.variable_scope('conv1d_2') as conv1d_scope:
                self.shared_conv1d_2 = []
                self.shared_out = []
                weights = tf.get_variable('weights',initializer=tf.truncated_normal(shape=[3,8,16],dtype=tf.float32,stddev=0.01))
                bias = tf.get_variable('bias',initializer=tf.constant(0.0,dtype=tf.float32,shape=[16]))

                if FLAGS.mode == 'train' or FLAGS.mode == 'test':
                    for i in range(self._task_num):
                        conv1d_tmp = tf.nn.conv1d(self.shared_out_1[i],weights,stride=1,padding='SAME')
                        out_tmp = tf.nn.relu(tf.nn.bias_add(conv1d_tmp,bias)) # [batchsize, 16, source_dim]
                        self.shared_conv1d_2.append(conv1d_tmp)
                        self.shared_out.append(out_tmp)
                elif FLAGS.mode == 'finetune':
                    conv1d_tmp = tf.nn.conv1d(self.out_tmp,weights,stride=1,padding='SAME')
                    self.out = tf.nn.relu(tf.nn.bias_add(conv1d_tmp,bias)) # [batchsize, 16, source_dim]
                # tf.summary.histogram(conv1d_scope.name+"weights",weights)
                # tf.summary.histogram(conv1d_scope.name+"bias",bias)

        if FLAGS.mode == 'train' or FLAGS.mode == 'test':
            for task_no in range(self._task_num):
                self.build_task_graph(task_no)
        elif FLAGS.mode == 'finetune':
            self.build_task_graph(task_no='new')
    def build_task_graph(self,task_no):
        with tf.variable_scope('task_{}'.format(task_no)) as scope:
            # convolution layer
            with tf.variable_scope('conv1d') as conv1d_scope:
                weights = tf.get_variable('weights',initializer=tf.truncated_normal(shape=[3,16,16],dtype=tf.float32,stddev=0.01))
                bias = tf.get_variable('bias',initializer=tf.constant(0.01,dtype=tf.float32,shape=[16]))
                if FLAGS.mode == 'train' or FLAGS.mode == 'test':
                    conv1d = tf.nn.conv1d(self.shared_out[task_no],weights,stride=1,padding='SAME')
                elif FLAGS.mode == 'finetune':
                    conv1d = tf.nn.conv1d(self.out,weights,stride=1,padding='SAME')
                    tf.summary.histogram(conv1d_scope.name+"/weights",weights)
                    tf.summary.histogram(conv1d_scope.name+"/bias",bias)
                task_out = tf.nn.relu(tf.nn.bias_add(conv1d,bias))
            with tf.variable_scope('fc1') as fc1_scope:
                shape = int(np.prod(task_out.get_shape()[1:]))
                weights = tf.get_variable('weights',initializer=tf.truncated_normal(shape=[shape,64],dtype=tf.float32,stddev=0.01))
                bias = tf.get_variable('bias',initializer=tf.constant(0.01,shape=[64],dtype=tf.float32))
                task_out_flat = tf.reshape(task_out,shape=[-1,shape])
                fc1 = tf.matmul(task_out_flat, weights)
                fc1_out = tf.nn.relu(tf.nn.bias_add(fc1,bias))
                fc1_out_dp = tf.nn.dropout(x=fc1_out,keep_prob=self.keep_prob)
                if FLAGS.mode == "finetune":
                    tf.summary.histogram(fc1_scope.name+"/weights",weights)
                    tf.summary.histogram(fc1_scope.name+"/bias",bias)
            with tf.variable_scope('fc2') as fc2_scope:
                weights = tf.get_variable('weights',initializer=tf.truncated_normal(shape=[64,FLAGS.num_classes],dtype=tf.float32,stddev=0.01))
                bias = tf.get_variable('bias',initializer=tf.constant(0.01,shape=[FLAGS.num_classes],dtype=tf.float32))
                task_out = tf.nn.bias_add(tf.matmul(fc1_out_dp, weights),bias)
                if FLAGS.mode == 'train' or FLAGS.mode == 'test':
                    self.task_outs.append(task_out)
                    self.task_outs_prob.append(tf.nn.softmax(task_out))
                elif FLAGS.mode == 'finetune':
                    self.task_out_prob = tf.nn.softmax(task_out)
                    self.task_out = task_out
                    tf.summary.histogram(fc2_scope.name+"/weights",weights)
                    tf.summary.histogram(fc2_scope.name+"/bias",bias)
    def add_losses(self,use_focal=False):
        for task_no in range(self._task_num):
            label_loss = self.add_loss(name=task_no,labels=self.ys[task_no],logits=self.task_outs[task_no],use_focal_loss=use_focal)
            self.losses.append(label_loss)
            tf.summary.scalar("task{}_loss".format(task_no),label_loss)
    def add_loss(self,name='new',labels=None,logits=None,use_focal_loss=False):
        """

        :param name:
        :param labels: 真实标记
        :param logits: 预测标记
        :param focal_loss: 是否使用Focal Loss
        :return:
        """
        if use_focal_loss:
            label_loss = tf.reduce_mean(focal_loss(onehot_labels=labels,cls_preds=logits,task_no=name))
        else:
            label_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

        return label_loss
    def add_joinly_loss(self):
        # 计算联合损失
        self.joinly_loss = tf.reduce_mean(tf.add_n(self.losses))
        tf.summary.scalar("joinly_loss",self.joinly_loss)

    def tf_confusion_metrics(self,labels, logits):
        """
        compute f1 score
        :param labels:
        :param logits:
        :return:
        """
        predictions = tf.argmax(logits, 1)
        actuals = tf.argmax(labels, 1)

        ones_like_actuals = tf.ones_like(actuals)
        zeros_like_actuals = tf.zeros_like(actuals)
        ones_like_predictions = tf.ones_like(predictions)
        zeros_like_predictions = tf.zeros_like(predictions)

        tp_op = tf.reduce_sum(tf.cast(tf.logical_and(
                                tf.equal(actuals, ones_like_actuals),
                                tf.equal(predictions, ones_like_predictions)),"float"))

        tn_op = tf.reduce_sum(tf.cast(tf.logical_and(
                    tf.equal(actuals, zeros_like_actuals),
                    tf.equal(predictions, zeros_like_predictions)),"float"))

        fp_op = tf.reduce_sum(tf.cast(tf.logical_and(
                    tf.equal(actuals, zeros_like_actuals),
                    tf.equal(predictions, ones_like_predictions)),"float"))

        fn_op = tf.reduce_sum(tf.cast(tf.logical_and(
                    tf.equal(actuals, ones_like_actuals),
                    tf.equal(predictions, zeros_like_predictions)),"float"))

        epsion = 1e-6
        tpr = tp_op / (tp_op + fn_op + epsion)
        fpr = fp_op / (tp_op + fn_op + epsion)

        recall = tpr
        precision = tp_op / (tp_op+fp_op+epsion)

        f1_score = (2 * (precision * recall)) / (precision + recall + epsion)

        return f1_score
    def add_f1_scores(self):
        for task_no in range(self._task_num):
            f1 = self.tf_confusion_metrics(labels=self.ys[task_no],logits=self.task_outs_prob[task_no])
            self.f1_scores.append(f1)
    def add_accuracies(self):
        for task_no in range(self._task_num):
            correct_prediction = tf.equal(tf.argmax(self.task_outs[task_no], 1), tf.argmax(self.ys[task_no], 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.accuracies.append(accuracy)
            tf.summary.scalar("task{}_acc".format(task_no),accuracy)
    def add_accuracy(self,name='new'):
        correct_prediction = tf.equal(tf.argmax(self.task_out, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    def add_joinly_acc(self):
        pass
    def add_trains(self):
        for task_no in range(self._task_num):
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            with tf.name_scope('train_op_{}'.format(task_no)) as scope:
                global_step = tf.Variable(0, trainable=False)
                train_op = optimizer.minimize(self.losses[task_no],global_step=global_step)
                self.train_ops.append(train_op)

    def add_train(self,name='new'):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        #  选择需要训练的变量
        all_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        to_do_train_vars = []
        for w in all_var:
            if 'shared_weights' in w.name: continue
            to_do_train_vars.append(w)
        with tf.name_scope('train_op_{}'.format(name)) as scope:
            global_step = tf.Variable(0, trainable=False)
            self.train_op = optimizer.minimize(self.loss, global_step=global_step,var_list=to_do_train_vars)

    def train_joinly(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        with tf.name_scope('train_op_joinly') as scope:
            global_step = tf.Variable(0, trainable=False)
            self.train_op_joinly = optimizer.minimize(self.joinly_loss,global_step=global_step)
    def build_graph(self,):
        if FLAGS.mode == 'train' or FLAGS.mode == 'test':
            self.add_placeholders()
            self.inference()
            self.add_losses(use_focal=True)
            self.add_joinly_loss()
            self.add_f1_scores()
            self.add_accuracies()
            self.add_joinly_acc()
            if FLAGS.train_mode == 'respective':
                self.add_trains()
            elif FLAGS.train_mode == 'join':
                self.train_joinly()
        elif FLAGS.mode == 'finetune':
            self.add_placeholder_x()
            self.add_placeholder_y()
            self.inference()
            self.loss = self.add_loss(name='new',labels=self.y,logits=self.task_out)
            self.add_accuracy(name='new')
            self.add_train(name='new')
            tf.summary.scalar('Loss',self.loss)
            tf.summary.scalar("task{}_acc".format("new"), self.accuracy)

    def fetch_task_no(self):
        rand = np.random.rand()
        intervals = 1/self._task_num
        rand_task_no = rand // intervals
        return int(rand_task_no)
    def fit(self,batches_train=[],batches_test=[],train_mode=FLAGS.train_mode,initialize=False):
        """
        根据训练模式分别进行训练
        :param batches_train: [[data for each task],[label for each task]]
        :param batches_test:  same as train
        :param train_mode:  train, test or finetune
        :param initialize:
        :return:
        """
        tasks_num = self._task_num
        X_batch_generator_train = batches_train[0]
        Y_batch_generator_train = batches_train[1]
        X_batch_generator_test = batches_test[0]
        Y_batch_generator_test = batches_test[1]
        # assert len(X_batch_generator_train) == self._task_num
        # assert len(X_batch_generator_test) == self._task_num
        #var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='shared_weights')
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        print("fitting the model with train mode {} and mode {}".format(train_mode,FLAGS.mode))
        start_time = datetime.datetime.now()
        saver = tf.train.Saver(max_to_keep=5,var_list=var_list)  # var_list 表示需要训练的权重
        if train_mode == 'respective':  #分开进行训练
            with tf.Session() as sess:
                if initialize: self.check_restore_parameters(sess,saver)
                train_writer = tf.summary.FileWriter(logdir=FLAGS.log_dir,graph=sess.graph)
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                coord_list = [tf.train.Coordinator() for _ in range(self._task_num)]
                threads_list = [tf.train.start_queue_runners(coord=i) for i in coord_list]
                should_stop = [False for _ in range(self._task_num)]
                #coord = tf.train.Coordinator()
                #threads = tf.train.start_queue_runners(coord=coord)

                iter = 0
                iter_tasks = [0 for _ in range(tasks_num)]

                tasks_loss = [[] for _ in range(tasks_num)]
                tasks_global_loss = [[] for _ in range(tasks_num)]
                tasks_loss_sum = [0 for _ in range(tasks_num)]
                tasks_acc = [[] for _ in range(tasks_num)]
                tasks_acc_global = [[] for _ in range(tasks_num)]
                tasks_f1 = [[] for _ in range(tasks_num)]
                tasks_f1_global = [[] for _ in range(tasks_num)]
                n_iteration = FLAGS.Iteration
                def coord_list_shouldnt_stop(coord_list):
                    for i,coord in enumerate(coord_list):
                        if coord.should_stop():
                            should_stop[i] = True
                            coord.request_stop()
                    if False in should_stop:
                        return True
                    return False
                try:
                    while (coord_list_shouldnt_stop(coord_list) and iter < n_iteration):
                        iter += 1
                        if iter %500 == 1: print("iter:{}/{}".format(iter,n_iteration))
                        merged = tf.summary.merge_all()

                        # 选择哪个task需要训练
                        train_task_no = self.fetch_task_no()
                        assert train_task_no < self._task_num
                        iter_tasks[train_task_no] += 1
                        tmp_x, tmp_y = sess.run([X_batch_generator_train[train_task_no],Y_batch_generator_train[train_task_no]])
                        feed_dict = {self.xs[train_task_no]:tmp_x,
                                     self.ys[train_task_no]:tmp_y}
                        _, loss,y_true,y_pred, acc,f1 = sess.run([self.train_ops[train_task_no], self.losses[train_task_no],
                                                 self.ys[train_task_no],self.task_outs_prob[train_task_no],
                                                 self.accuracies[train_task_no],self.f1_scores[train_task_no]]
                                                              , feed_dict=feed_dict)
                        print("     Iter - {}, task - {}(ALL:{})".format(iter,train_task_no,iter_tasks[train_task_no]),acc,f1)
                        #_, loss, y_true, y_pred, acc, f1,summary = sess.run(
                        #    [self.train_ops[train_task_no], self.losses[train_task_no],
                        #     self.ys[train_task_no], self.task_outs_prob[train_task_no],
                        #     self.accuracies[train_task_no], self.f1_scores[train_task_no],merged]
                        #    , feed_dict=feed_dict)

                        tasks_loss[train_task_no].append(loss)
                        tasks_loss_sum[train_task_no] += loss
                        tasks_global_loss[train_task_no].append(tasks_loss_sum[train_task_no]/iter_tasks[train_task_no])
                        tasks_acc[train_task_no].append(acc)
                        tasks_acc_global[train_task_no].append(sum(tasks_acc[train_task_no])/iter_tasks[train_task_no])
                        tasks_f1[train_task_no].append(f1)
                        tasks_f1_global[train_task_no].append(sum(tasks_f1[train_task_no])/iter_tasks[train_task_no])
                        if iter_tasks[train_task_no]%250 == 1:
                            cur_time = datetime.datetime.now()
                            #print("executing time: {} mins".format((cur_time-start_time).seconds/60))
                            #print("     Train--task{}_loss: {}, acc: {}, f1: {}".format(train_task_no,loss,acc,f1))
                            # 在testdata上再尝试一下
                            #print("rate: {}".format(sum(y_true[:,1]/len(y_true))))
                            tmp_x, tmp_y = sess.run(
                                [X_batch_generator_test[train_task_no], Y_batch_generator_test[train_task_no]])
                            feed_dict = {self.xs[train_task_no]: tmp_x,
                                         self.ys[train_task_no]: tmp_y}
                            # _, loss, acc,summary = sess.run([self.train_op[train_task_no],self.losses[train_task_no],
                            #                    self.accuracies[train_task_no],merged],feed_dict=feed_dict)
                            loss,acc,f1 = sess.run([self.losses[train_task_no],self.accuracies[train_task_no],
                                                    self.f1_scores[train_task_no]], feed_dict=feed_dict)
                            print("     Test--task{}_loss: {}, acc: {}, f1: {}".format(train_task_no,loss,acc,f1))
                        if iter%1000==0:
                            saver.save(sess=sess, save_path=FLAGS.checkpoints_dir, global_step=iter)
                        #train_writer.add_summary(summary=summary,global_step=iter)

                except tf.errors.OutOfRangeError:
                    print('Reading Done!')
                finally:
                    print('Executing every task Done!!')


                coord_list[-1].join(threads_list[-1])
                #coord.join(threads)

                sess.close()
                print("ploting...")
                plot(losses=tasks_global_loss, acc=tasks_f1_global)
        elif train_mode == 'join':      #联合训练
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                merged = tf.summary.merge_all()
                train_writer = tf.summary.FileWriter(logdir=FLAGS.log_dir,graph=sess.graph)
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)
                loss_all = 0
                iter = 0
                losses = []
                global_loss = []

                acc_all = 0
                global_acc = []
                try:
                    while iter < 1000:
                        if iter%50 == 0: print("{}/{}".format(iter+1,1000))
                        iter += 1
                        X_batches = sess.run(X_batch_generator_train)
                        Y_batches = sess.run(Y_batch_generator_train)
                        feed_dict_X = {kk:vv for kk,vv in zip(self.xs, X_batches)}
                        feed_dict_Y = {kk:vv for kk,vv in zip(self.ys, Y_batches)}
                        feed_dict = {}
                        feed_dict.update(feed_dict_X)
                        feed_dict.update(feed_dict_Y)
                        _, loss,summary = sess.run([self.train_op_joinly, self.joinly_loss,merged],feed_dict=feed_dict)
                        losses.append(loss)
                        loss_all += loss
                        global_loss.append(loss_all/iter)
                        if iter%20 == 0: print(loss_all/iter)

                        train_writer.add_summary(summary=summary)
                except tf.errors.OutOfRangeError:
                    print("Reading Done!")
                finally:
                    coord.request_stop()
                coord.join(threads)
                sess.close()
                plot([losses, global_loss],['loss','global_loss'])
    def check_restore_parameters(self,sess):
        ckpt_path = tf.train.latest_checkpoint(checkpoint_dir="./checkpoints")
        meta_path_restore = "{}.meta".format(ckpt_path)
        model_path_restore = os.path.join(ckpt_path)

        # ckpt = tf.train.get_checkpoint_state(checkpoint_dir="./checkpoints")
        shared_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='shared_weights')
        saver_restore = tf.train.Saver(var_list=shared_weights)
        # saver_restore = tf.train.import_meta_graph(meta_path_restore)
        if ckpt_path:
            print("Initialing the network with previous parameters...")
            saver_restore.restore(sess,ckpt_path)
        else:
            print("There are no checkpoints to initialize the network!")
    def refit(self,train = [],test = []):
        """
        use new data to refit the model
        :param train: [x_batch,y_batch]
        :param test:  [x_batch,y_batch]
        :return:
        """
        train_x, train_y = train[0],train[1]
        test_x, test_y = test[0], test[1]
        saver = tf.train.Saver(max_to_keep=5)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            self.check_restore_parameters(sess)
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(logdir=FLAGS.log_dir_newTask,graph=sess.graph)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            loss_all = 0
            acc_all = 0
            iter = 0
            losses = []
            global_loss = []
            global_acc = []
            try:
                while iter < FLAGS.Iteration:
                    if iter % 2000 == 0: print("{}/{}".format(iter + 1, FLAGS.Iteration))
                    iter += 1
                    X_batches = sess.run(train_x)
                    Y_batches = sess.run(train_y)
                    feed_dict = {self.x:X_batches, self.y:Y_batches}
                    _, loss, acc, summary,y_true,y_pred = sess.run([self.train_op, self.loss,self.accuracy,merged, self.y,self.task_out_prob], feed_dict=feed_dict)
                    losses.append(loss)
                    loss_all += loss
                    acc_all += acc
                    global_loss.append(loss_all / iter)
                    global_acc.append(acc_all / iter)
                    if iter % 500 == 0:
                        print("Loss: {}, Accuraccy: {}".format(loss_all / iter, acc))
                        print("rate: {}".format(sum(y_true[:,1])/len(y_true)))
                        for i,j in zip(y_true[:5], y_pred[:5]):
                            print("true: {}, pred: {}".format(i[1], j[1]))
                        saver.save(sess,save_path=FLAGS.checkpoints_dir_newTask,global_step=iter)
                    train_writer.add_summary(summary=summary,global_step=iter)
            except tf.errors.OutOfRangeError:
                print("Reading Done!")
            finally:
                coord.request_stop()
            coord.join(threads)
            sess.close()
            plot(losses=[global_loss],acc=[global_acc])
    def predict(self,batches_test,train_mode=FLAGS.train_mode):
        print("prediction program starting...")
        X_batch_generator_test = batches_test[0]
        Y_batch_generator_test = batches_test[1]
        tasks_num = len(X_batch_generator_test)
        task_predictions = [[] for _ in range(tasks_num)]
        if train_mode == 'respective':  # 分开进行训练
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                saver = tf.train.Saver(weights)
                ckpt_path = tf.train.latest_checkpoint(checkpoint_dir='./checkpoints')
                meta_path_restore = "{}.meta".format(ckpt_path)
                model_path_restore = "{}.data-00000-of-00001".format(ckpt_path)
                saver_restore = tf.train.import_meta_graph(meta_path_restore)
                #saver_restore.restore(sess,model_path_restore)
                saver.restore(sess,ckpt_path)

                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)
                iter = 0
                for task_no in range(tasks_num):
                    try:
                        while True:
                            iter += 1
                            if iter % 1000 == 0: print("iter:{}".format(iter))

                            tmp_x, tmp_y = sess.run(
                                [X_batch_generator_test[task_no], Y_batch_generator_test[task_no]])
                            feed_dict = {self.xs[task_no]: tmp_x,
                                         self.ys[task_no]: tmp_y}
                            y_true, y_pred= sess.run([self.ys[task_no], self.task_outs_prob[task_no]],feed_dict=feed_dict)
                            task_predictions[task_no].append([y_true[0][1],y_pred[0][1]])
                    except tf.errors.OutOfRangeError:
                        print('Reading Done!')
                    finally:
                        coord.request_stop()

                    coord.join(threads)
                sess.close()
            # 将预测结果保存下来
            for task_no in range(tasks_num):
                results = pd.DataFrame(data=np.asarray(task_predictions[task_no]),columns=['true','prediction'])
                results.to_csv(os.path.join(FLAGS.results_dir,"task{}_prediction_label.csv".format(task_no)),index_label='id')

def one_hot_encoding(y_batch):
    batch_size = tf.size(y_batch)
    y_test_batch = tf.expand_dims(y_batch, axis=1)
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated = tf.concat([indices, y_test_batch], 1)
    one_hot = tf.sparse_to_dense(concated, tf.stack([batch_size, 2]), 1.0, 0.0)
    return one_hot

def train_model(model,task_list = [0,1,2],initialize=False):

    # get train data and validation data
    train_dataset, test_dataset = get_data_batch(task_list=task_list,batchsize=FLAGS.batch_size,n_epoch=FLAGS.num_epochs)
    X_batch_train = []
    Y_batch_train = []
    X_batch_test = []
    Y_batch_test = []

    for i,i_x_batch_train in enumerate(train_dataset[0]):
        # x_train_batch_feed = tf.reshape(i_x_batch_train, shape=[-1, FLAGS.input_dim, 1])
        x_train_batch_feed = i_x_batch_train
        X_batch_train.append(x_train_batch_feed)
        Y_batch_train.append(train_dataset[1][i])

    for i,i_x_batch_test in enumerate(test_dataset[0]):
        x_test_batch_feed = i_x_batch_test
        X_batch_test.append(x_test_batch_feed)
        Y_batch_test.append(test_dataset[1][i])

    model.fit(batches_train=[X_batch_train,Y_batch_train],batches_test=[X_batch_test,Y_batch_test],initialize=initialize)

def test_model(task_no=TASK_LIST):
    _, test_dataset = get_data_batch(task_list=TASK_LIST,batchsize=1,n_epoch=1)
    X_batch_test = []
    Y_batch_test = []
    for i,i_x_batch_test in enumerate(test_dataset[0]):
        x_test_batch_feed = i_x_batch_test
        X_batch_test.append(x_test_batch_feed)
        Y_batch_test.append(test_dataset[1][i])

    model.predict(batches_test=[X_batch_test, Y_batch_test])
def finetune(task_no = 3):

    train_dataset, test_dataset = get_data_batch(task_list=[task_no],batchsize=FLAGS.batch_size,n_epoch=FLAGS.num_epochs)
    #x_batch_train = tf.reshape(train_dataset[0][0], shape=[-1, FLAGS.input_dim, 1])
    x_batch_train = train_dataset[0][0]
    y_batch_train = train_dataset[1][0]
    #x_batch_test = tf.reshape(test_dataset[0][0], shape=[-1, FLAGS.input_dim, 1])
    x_batch_test = test_dataset[0][0]
    y_batch_test = test_dataset[1][0]
    model.refit(train=[x_batch_train,y_batch_train], test=[x_batch_test,y_batch_test])

if __name__ == '__main__':

    model = MTL(FLAGS.mode)
    model.build_graph()
    print('build graph success!')
    task_list = TASK_LIST

    if FLAGS.mode == 'train':
        # training a model
        train_model(model,task_list=task_list,initialize=False)
    elif FLAGS.mode == 'test':
        # test a model
        test_model()
    elif FLAGS.mode == 'finetune':
        # Now a new task comming! Finetune it
        finetune(task_no=3)