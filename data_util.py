import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import argparse
import pandas as pd
import numpy as np
import os
from xgboost import XGBClassifier
data_dir = "./data"

parser = argparse.ArgumentParser()
parser.add_argument('--n_samples',help='number of samples',default=8000,type=int)
parser.add_argument('--pos_rate',help='positive samples rate',default=0.45,type=float)
parser.add_argument('--task_num',help='please input your task num',default=4,type=int)
parser.add_argument('--input_dim',help='please input MTL input dim',default=20,type=int)
parser.add_argument('--embedding',help='do you want to use tree embedding?',default=0,type=int)
args = parser.parse_args()
if args.embedding == 1:
    append = "_embedding"
elif args.embedding == 0:
    append = ""

def data_generation(task_no=1):
    np.random.seed(task_no)

    train_file = os.path.join(data_dir,"task{}_train.csv".format(task_no))
    test_file = os.path.join(data_dir,"task{}_test.csv".format(task_no))

    if task_no == 1:
        n_samples = args.n_samples * 5
    else:
        n_samples = args.n_samples
    X, y = make_classification(n_samples=n_samples, weights=[1 - args.pos_rate, args.pos_rate])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    n_samples, n_cols = X_train.shape
    # 将数据保存下来
    columns = ["f_{}".format(i+1) for i in range(n_cols)]
    train_df = pd.DataFrame(data=np.hstack((y_train.reshape(-1,1),X_train)),columns=['label']+columns,dtype=np.float32)
    train_df['label'] = train_df['label'].astype(np.int32)
    train_df.to_csv(train_file,index_label='id')
    test_df = pd.DataFrame(data=np.hstack((y_test.reshape(-1,1),X_test)),columns=['label']+columns,dtype=np.float32)
    test_df['label'] = test_df['label'].astype(np.int32)
    test_df.to_csv(test_file,index_label='id')

def read_data(file_queue):
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(file_queue)
    if args.embedding == 1:
        defaults = [[0]]    # id & label
        defaults_data = [[0.] for _ in range(args.input_dim)]
        defaults.extend(defaults_data)
        defaults.append([0])
        data_all = tf.decode_csv(value, defaults)
        id = data_all[0]
        label = data_all[-1]
        data = data_all[1:-1]
    elif args.embedding == 0:
        defaults = [[0],[0]]  # id & label
        defaults_data = [[0.] for _ in range(args.input_dim)]
        defaults.extend(defaults_data)
        data_all = tf.decode_csv(value, defaults)
        id = data_all[0]
        label = data_all[1]
        data = data_all[2:]
    return data, label
def create_pipeline(filename, batch_size=64, num_epoches=1):
    file_queue = tf.train.string_input_producer([filename],num_epochs=num_epoches)
    example, label = read_data(file_queue)
    min_after_dequeue = 1000
    capacity = min_after_dequeue + batch_size
    example_batch, label_batch = tf.train.shuffle_batch([example, label],batch_size=batch_size,capacity=capacity,
                                                        min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch

def one_hot_encoding(y_batch):
    batch_size = tf.size(y_batch)
    y_test_batch = tf.expand_dims(y_batch, axis=1)
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated = tf.concat([indices, y_test_batch], 1)
    one_hot = tf.sparse_to_dense(concated, tf.stack([batch_size, 2]), 1.0, 0.0)
    return one_hot

def get_data_batch(task_list=list(range(args.task_num)),batchsize=64,n_epoch=10):
    """
    根据task_list里面的task no 返回每个task的数据
    :param task_list: [0,1,2]
    :return: [[task1_train_data, task2_train_data],[task1_train_label,task2_train_label]],
             [[task1_test_data, task2_test_data],[task1_test_label,task2_test_label]]
    """
    task_train_data = []
    task_train_label = []
    task_test_data = []
    task_test_label = []
    for task_no in task_list:
        train_file_path = os.path.join(data_dir, "task{}_train{}.csv".format(task_no,append))
        assert os.path.exists(train_file_path),'{}不存在！'.format(train_file_path)
        test_file_path = os.path.join(data_dir,"task{}_test{}.csv".format(task_no,append))
        assert os.path.exists(test_file_path),"{}不存在！".format(test_file_path)

        x_train_batch, y_train_batch = create_pipeline(filename=train_file_path,batch_size=batchsize,num_epoches=n_epoch)
        x_test_batch, y_test_batch = create_pipeline(filename=test_file_path,batch_size=batchsize,num_epoches=n_epoch)

        y_train_batch_one_hot = one_hot_encoding(y_train_batch)
        y_test_batch_one_hot = one_hot_encoding(y_test_batch)
        task_train_data.append(x_train_batch)
        task_test_data.append(x_test_batch)
        task_train_label.append(y_train_batch_one_hot)
        task_test_label.append(y_test_batch_one_hot)
    return [task_train_data, task_train_label], [task_test_data,task_test_label]
def TreeEmbedding():
    for task_no in range(args.task_num):
        train_file = os.path.join(data_dir, "task{}_train.csv".format(task_no))
        test_file = os.path.join(data_dir, "task{}_test.csv".format(task_no))
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)
        ID = 'id'
        target = 'label'
        columns = [x for x in train_data.columns if x not in [ID,target]]
        X_train = train_data[columns]
        y_train = train_data[target]
        X_test = test_data[columns]
        y_test = test_data[target]
        xgb = XGBClassifier(n_estimators=500,max_depth=4,subsample=0.6,colsample_bytree=0.6)
        xgb.fit(X_train, y_train)
        X_train_embedding = xgb.apply(X_train)
        X_test_embedding = xgb.apply(X_test)
        field_num = X_train_embedding.shape[1]
        field_id = ["t{}".format(i+1) for i in range(field_num)]

        leaves_num = sum([max(X_train_embedding[:,i]) for i in range(field_num)])
        print("leaves num: {}".format(leaves_num))
        with open('./data/parameters.conf','w') as file:
            file.write("leaves_num:{}".format(leaves_num))
        train_data_path = train_file.replace(".csv","_embedding.csv")
        test_data_path = test_file.replace(".csv","_embedding.csv")

        print('saving RandomTreesEmbedding datasets...')
        # 将训练集和测试集的Tree Embedding 保存下来
        X_train_embedding_df = pd.DataFrame(data=X_train_embedding,columns=field_id)
        for field_name in field_id:
            X_train_embedding_df[field_name] = (X_train_embedding_df[field_name]-X_train_embedding_df[field_name].min())/\
                                               (X_train_embedding_df[field_name].max()-X_train_embedding_df[field_name].min())
        # 增加label列
        X_train_embedding_df['label'] = y_train
        X_train_embedding_df.to_csv(train_data_path,index_label='id')

        # 保存测试集
        X_test_embedding_df = pd.DataFrame(data=X_test_embedding, columns=field_id)
        for field_name in field_id:
            X_test_embedding_df[field_name] = (X_test_embedding_df[field_name] - X_test_embedding_df[field_name].min()) / \
                                               (X_test_embedding_df[field_name].max() - X_test_embedding_df[field_name].min())

        X_test_embedding_df['label'] = y_test
        X_test_embedding_df.to_csv(test_data_path,index_label='id')


if __name__ == '__main__':
    for task_no in range(args.task_num):
        data_generation(task_no=task_no)
    #TreeEmbedding()