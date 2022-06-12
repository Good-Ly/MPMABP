import os
import  keras
import pickle
import numpy as np
import tensorflow as tf
import keras.backend as K
from pathlib import Path
from keras.regularizers import l2
from keras.optimizers import Adam, SGD,Adagrad
from keras.layers.wrappers import Bidirectional
from sklearn.model_selection import train_test_split,KFold
from keras.models import load_model,Model, Sequential,model_from_json
#from keras.layers import Input, Embedding, Convolution1D, MaxPooling1D, Concatenate, Dropout
from keras.layers import *
from sklearn.metrics import confusion_matrix,roc_auc_score,matthews_corrcoef,roc_curve,auc
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score,precision_recall_curve


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KERAS_BACKEND']='tensorflow'
np.random.seed(101)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

K.tensorflow_backend.set_session(tf.Session(config=config))

# -*- coding: utf-8 -*-
# @FileName: model.py
# @Software: PyCharm

#**************************************************************************************************#
# @Author  : Wending Tang
def scores(y_test,y_pred,th=0.5):
    y_predlabel=[(0 if item<th else 1) for item in y_pred]
    tn,fp,fn,tp=confusion_matrix(y_test,y_predlabel).flatten()
    SPE=tn*1./(tn+fp)
    MCC=matthews_corrcoef(y_test,y_predlabel)
    Recall=recall_score(y_test, y_predlabel)
    Precision=precision_score(y_test, y_predlabel)
    F1=f1_score(y_test, y_predlabel)
    Acc=accuracy_score(y_test, y_predlabel)
    AUC=roc_auc_score(y_test, y_pred)
    precision_aupr, recall_aupr, _ = precision_recall_curve(y_test, y_pred)
    AUPR = auc(recall_aupr, precision_aupr)
    return [Recall,SPE,Precision,F1,MCC,Acc,AUC,AUPR,tp,fn,tn,fp]

def Aiming(y_hat, y):
    '''
    the “Aiming” rate (also called “Precision”) is to reflect the average ratio of the
    correctly predicted labels over the predicted labels; to measure the percentage
    of the predicted labels that hit the target of the real labels.
    '''

    n, m = y_hat.shape

    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        sorce_k += intersection / sum(y_hat[v])
    return sorce_k / n


def Coverage(y_hat, y):
    '''
    The “Coverage” rate (also called “Recall”) is to reflect the average ratio of the
    correctly predicted labels over the real labels; to measure the percentage of the
    real labels that are covered by the hits of prediction.
    '''

    n, m = y_hat.shape

    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        sorce_k += intersection / sum(y[v])

    return sorce_k / n


def Accuracy(y_hat, y):
    '''
    The “Accuracy” rate is to reflect the average ratio of correctly predicted labels
    over the total labels including correctly and incorrectly predicted labels as well
    as those real labels but are missed in the prediction
    '''

    n, m = y_hat.shape

    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        sorce_k += intersection / union
    return sorce_k / n


def AbsoluteTrue(y_hat, y):
    '''
    same
    '''

    n, m = y_hat.shape
    sorce_k = 0
    for v in range(n):
        if list(y_hat[v]) == list(y[v]):
            sorce_k += 1
    return sorce_k/n


def AbsoluteFalse(y_hat, y):
    '''
    hamming loss
    '''

    n, m = y_hat.shape

    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v,h] == 1 or y[v,h] == 1:
                union += 1
            if y_hat[v,h] == 1 and y[v,h] == 1:
                intersection += 1
        sorce_k += (union-intersection)/m
    return sorce_k/n


def evaluate(y_hat, y):
    aiming = Aiming(y_hat, y)
    coverage = Coverage(y_hat, y)
    accuracy = Accuracy(y_hat, y)
    absolute_true = AbsoluteTrue(y_hat, y)
    absolute_false = AbsoluteFalse(y_hat, y)
    return aiming, coverage, accuracy, absolute_true, absolute_false

def catch(data, label):
    # preprocessing label and data
    l = len(data)
    chongfu = 0
    for i in range(l):
        ll = len(data)
        idx = []
        each = data[i]
        j = i + 1
        bo = False
        while j < ll:
            if (data[j] == each).all():
                label[i] += label[j]
                idx.append(j)
                bo = True
            j += 1
        t = [i] + idx
        if bo:
            #print(t)
            chongfu += 1
            #print(data[t[0]])
            #print(data[t[1]])
        data = np.delete(data, idx, axis=0)
        label = np.delete(label, idx, axis=0)

        if i == len(data)-1:
            break
    print('total number of the same data: ', chongfu)

    return data, label

def predict(X_test, y_test, thred, para, weights, jsonFiles, h5_model, dir):

    # with open('test_true_label.pkl', 'wb') as f:
    #     pickle.dump(y_test, f)

    adam = Adam(lr=para['learning_rate']) # adam optimizer
    for ii in range(0, len(weights)):
        # 1.loading weight and structure (model)

        # json_file = open('MFBPP_model/' + jsonFiles[i], 'r')
        # model_json = json_file.read()
        # json_file.close()
        # load_my_model = model_from_json(model_json)
        # load_my_model.load_weights('MFBPP_model/' + weights[i])
        # load_my_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

        h5_model_path = os.path.join(dir, h5_model[ii])
        #print(h5_model_path)
        load_my_model = load_model(h5_model_path)
        print("Prediction is in progress")
        #print("ii:%d"%(ii))

        # 2.predict
        score = load_my_model.predict(X_test)

        "========================================"
        for i in range(len(score)):
            for j in range(len(score[i])):
                if score[i][j] < thred:
                    score[i][j] = 0
                else:
                    score[i][j] = 1
        a, b, c, d, e = evaluate(score, y_test)
        #print(a, b, c, d, e)
        "========================================"

        # 3.evaluation
        if ii == 0:
            score_label = score
        else:
            score_label += score

    score_label = score_label / len(h5_model)
     # data saving
    with open(os.path.join(dir, 'MLBP_prediction_prob.pkl'), 'wb') as f:
        pickle.dump(score_label, f)
    
    # getting prediction label
    for i in range(len(score_label)):
        for j in range(len(score_label[i])):
            if score_label[i][j] < thred: score_label[i][j] = 0
            else: score_label[i][j] = 1

    # data saving
    with open(os.path.join(dir, 'MLBP_prediction_label.pkl'), 'wb') as f:
        pickle.dump(score_label, f)
    

    # evaluation
    aiming, coverage, accuracy, absolute_true, absolute_false = evaluate(score_label, y_test)

    print("Prediction is done")
    print('aiming:', aiming)
    print('coverage:', coverage)
    print('accuracy:', accuracy)
    print('absolute_true:', absolute_true)
    print('absolute_false:', absolute_false)
    print('\n')
    out = dir
    Path(out).mkdir(exist_ok=True, parents=True)
    out_path2 = os.path.join(out, 'result_test.txt')
    with open(out_path2, 'w') as fout:
        fout.write('aiming:{}\n'.format(aiming))
        fout.write('coverage:{}\n'.format(coverage))
        fout.write('accuracy:{}\n'.format(accuracy))
        fout.write('absolute_true:{}\n'.format(absolute_true))
        fout.write('absolute_false:{}\n'.format(absolute_false))
        fout.write('\n')
    return aiming, coverage, accuracy, absolute_true, absolute_false


def GetSourceData(root, dir, lb):
    seqs = []
    print('\n')
    print('now is ', dir)
    file = '{}CD.txt'.format(dir)
    file_path = os.path.join(root, dir, file)

    with open(file_path) as f:
        for each in f:
            if each == '\n' or each[0] == '>':
                continue
            else:
                seqs.append(each.rstrip())

    # data and label
    label = len(seqs) * [lb]
    seqs_train, seqs_test, label_train, label_test = train_test_split(seqs, label, test_size=0.2, random_state=0)
    print('train data:', len(seqs_train))
    print('test data:', len(seqs_test))
    print('train label:', len(label_train))
    print('test_label:', len(label_test))
    print('total numbel:', len(seqs_train)+len(seqs_test))

    return seqs_train, seqs_test, label_train, label_test



def DataClean(data):
    max_len = 0
    for i in range(len(data)):
        st = data[i]
        # get the maximum length of all the sequences
        if(len(st) > max_len): max_len = len(st)

    return data, max_len



def PadEncode(data, max_len):

    # encoding
    amino_acids = 'XACDEFGHIKLMNPQRSTVWY'
    data_e = []
    for i in range(len(data)):
        length = len(data[i])
        elemt, st = [], data[i]
        for j in st:
            index = amino_acids.index(j)
            elemt.append(index)
        if length < max_len:
            elemt += [0]*(max_len-length)
        data_e.append(elemt)

    return data_e



def GetSequenceData(dirs, root):
    # getting training data and test data
    count, max_length = 0, 0
    tr_data, te_data, tr_label, te_label = [], [], [], []
    for dir in dirs:
        # 1.getting data from file
        tr_x, te_x, tr_y, te_y = GetSourceData(root, dir, count)
        count += 1

        # 2.getting the maximum length of all sequences
        tr_x, len_tr = DataClean(tr_x)
        te_x, len_te = DataClean(te_x)
        if len_tr > max_length: max_length = len_tr
        if len_te > max_length: max_length = len_te

        # 3.dataset
        tr_data += tr_x
        te_data += te_x
        tr_label += tr_y
        te_label += te_y


    # data coding and padding vector to the filling length
    traindata = PadEncode(tr_data, max_length)
    testdata = PadEncode(te_data, max_length)

    # data type conversion
    train_data = np.array(traindata)
    test_data = np.array(testdata)
    train_label = np.array(tr_label)
    test_label = np.array(te_label)

    return [train_data, test_data, train_label, test_label]



def GetData(path):
    dirs = ['AMP', 'ACP', 'ADP', 'AHP', 'AIP'] # functional peptides

    # get sequence data
    sequence_data = GetSequenceData(dirs, path)

    return sequence_data

dir = 'MFBPP_model'
Path(dir).mkdir(exist_ok=True)
model_path = dir
Path(model_path).mkdir(exist_ok=True)


# I.get sequence data
path = 'data' # data path
sequence_data = GetData(path)
# sequence data partitioning
tr_seq_data,te_seq_data,tr_seq_label,te_seq_label = \
    sequence_data[0],sequence_data[1],sequence_data[2],sequence_data[3]
#TrainAndTest(tr_seq_data, tr_seq_label, te_seq_data, te_seq_label)




tr_data = tr_seq_data
tr_label = tr_seq_label
te_data = te_seq_data
te_label = te_seq_label

train = [tr_data, tr_label]
test = [te_data, te_label]


X_train, y_train = train[0], train[1]

# data and label preprocessing
y_train = keras.utils.to_categorical(y_train)
X_train, y_train = catch(X_train, y_train)
y_train[y_train > 1] = 1


# disorganize
index = np.arange(len(y_train))
np.random.shuffle(index)
X_train = X_train[index]
y_train = y_train[index]

# train
length = X_train.shape[1]
out_length = y_train.shape[1]


test[1] = keras.utils.to_categorical(test[1]) # test_data
test[0], temp = catch(test[0], test[1])
temp[temp > 1] = 1
test[1] = temp #label

threshold = 0.5

test.append(threshold)
#**************************************************************************************************#
# @Author  : liyou
print(X_train.shape)
print(y_train.shape)
print(test[0].shape)
print(test[1].shape)

def MFBPP(length, out_length, para):
    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.002

    main_input = Input(shape=(length,), dtype='int64', name='main_input')

    x = Embedding(output_dim=ed, input_dim = 21, input_length=length, embeddings_initializer='uniform')(main_input)
    x = keras.layers.BatchNormalization()(x)
    a = Convolution1D(64, 3,activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    a = Bidirectional(CuDNNLSTM(32, return_sequences=True))(a)
    a = keras.layers.advanced_activations.LeakyReLU(alpha=0.3)(a)
    apool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(a)

    b = Convolution1D(64,5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    b = Bidirectional(CuDNNLSTM(32, return_sequences=True))(b)
    b = keras.layers.advanced_activations.LeakyReLU(alpha=0.3)(b)
    bpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(b)

    c = Convolution1D(64,8, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    c = Bidirectional(CuDNNLSTM(32, return_sequences=True))(c)
    c = keras.layers.advanced_activations.LeakyReLU(alpha=0.3)(c)
    cpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(c)

    d = Convolution1D(64, 10, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    d = keras.layers.advanced_activations.LeakyReLU(alpha=0.3)(d)
    d = Bidirectional(CuDNNLSTM(32, return_sequences=True))(d)
    dpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(d)
    
    e = Convolution1D(64,12, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    e = keras.layers.advanced_activations.LeakyReLU(alpha=0.3)(e)
    e = Bidirectional(CuDNNLSTM(32, return_sequences=True))(e)
    epool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(e)
    

    #x_batchnorm = keras.layers.BatchNormalization()(x)
    cnnrnn = Concatenate(axis=-1)([apool,bpool,cpool,dpool,epool,x])

    CNNRNN = Flatten()(cnnrnn)
    CNNRNN = Dense(64, activation='relu', name='dense1', W_regularizer=l2(l2value))(CNNRNN)
    CNNRNN = Dropout(dp)(CNNRNN)
    CNNRNN = Dense(128, activation='relu', name='dense2', W_regularizer=l2(l2value))(CNNRNN)

    output = Dense(out_length, activation='sigmoid', name='output', W_regularizer=l2(l2value))(CNNRNN)

    model = Model(inputs=main_input, output=output)
    adam = Adagrad(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    # model.summary()

    return model



ed = 100
ps = 3
fd = 64
dp = 0.5
lr = 0.001
para = {'embedding_dimension': ed, 'pool_size': ps, 'fully_dimension': fd,
        'drop_out': dp, 'learning_rate': lr}
model = MFBPP(length, out_length, para)

#model.fit(X_train, y_train, nb_epoch = 34, batch_size = 30, verbose = 2)#3
#each_model = os.path.join(model_path, 'Independently_tested_model' + '.h5')
#model.save(each_model)
weights = []
jsonFiles = []
h5_model = []

weights.append("Independently_tested_model.h5")
jsonFiles.append("Independently_tested_model.h5")
h5_model.append("Independently_tested_model.h5")

# step2:predict
aiming, coverage, accuracy, absolute_true, absolute_false = predict(test[0], test[1], test[2], para, weights, jsonFiles, h5_model, dir) # test[2] 阈值

num_model = 1
kf = KFold(n_splits=5 )
Aim = []
Cov = []
Acc = []
Abs_true = []
Abs_false = []
for train_index, test_index in kf.split(X_train):
    #model_for_5cros_val = MFBPP(length, out_length, para)
    print("********************     model %d  *****************"%(num_model))
    #model_for_5cros_val.fit(X_train[train_index], y_train[train_index], nb_epoch = 34, batch_size = 30, verbose=2)
    #each_model = os.path.join(model_path, "model"+ str(num_model) + '_5_cross_val.h5')
    #model_for_5cros_val.save(each_model)
    weights = []
    jsonFiles = []
    h5_model = []

    weights.append('model{}_5_cross_val.hdf5'.format(str(num_model)))
    jsonFiles.append('model{}_5_cross_val.json'.format(str(num_model)))
    h5_model.append('model{}_5_cross_val.h5'.format(str(num_model)))  
    aiming, coverage, accuracy, absolute_true, absolute_false = predict(X_train[test_index], y_train[test_index],test[2],para, weights, jsonFiles, h5_model, dir)
    Aim.append(aiming)
    Cov.append(coverage)
    Acc.append(accuracy)
    Abs_true.append(absolute_true)
    Abs_false.append(absolute_false)
    num_model += 1
std_aim = np.std(Aim)
std_Cov = np.std(Cov)
std_Acc = np.std(Acc)
std_abs_t = np.std(Abs_true)
std_abs_f = np.std(Abs_false)

print(r"Aim:%.3f，std: %.3f"%(np.mean(Aim),std_aim))
print(r"Cov:%.3f, std: %.3f"%(np.mean(Cov),std_Cov))
print(r"Acc:%.3f, std: %.3f"%( np.mean(Acc),std_Acc))
print(r"Abs_true:%.3f,std: %.3f"%(np.mean(Abs_true),std_abs_t))
print(r"Abs_false:%.3f,std:%.3f"%(np.mean(Abs_false),std_abs_f))
print("\n")
#**************************************************************************************************#
