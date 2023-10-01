import sys
import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold, KFold
from keras import Model
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Input, LSTM, Conv1D, Conv2D, Flatten, MaxPooling1D, Concatenate
from keras.utils import np_utils
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import StandardScaler

def crossValidationSSS(model, X, y, n_splits=10, test_size=0.2, train_size=0.8, random_state=42):
    kfold = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=train_size, random_state=random_state)
    results = []
    for train_idx, test_idx in kfold.split(X, y):
        X_t = X[train_idx]
        y_t = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]
        X_train, X_val, y_train, y_val = train_test_split(X_t, y_t, test_size=0.10, stratify=y_t)
        y_w = np.argmax(np.array(y_train), 1)
        unique, counts = np.unique(y_w, return_counts=True)
        class_weights = dict(zip(unique, counts))
        weights = compute_sample_weight(class_weights, y_w)
        early_stop = EarlyStopping(monitor='loss', patience=20, verbose=0)
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200, batch_size=10000, verbose=1, callbacks=[early_stop], shuffle=True, sample_weight=weights)
        y_pred = model.predict(X_test)
        results.append(evaluate(y_test, y_pred, weights))
    a = np.array(results)
    avg = np.mean(a, axis=0)
    std = np.std(a, axis=0)
    scores = ['accuracy', 'weighted precision', 'micro precision', 'weighted recall', 'micro recall']
    for elem in zip(scores, avg, std):
        print(elem)
    return results

def evaluate(y_test, y_pred, weights):
    y_pred_norm = []

    for elem in y_pred:
        line = [ 0 ] * len(elem)
        line[elem.tolist().index(max(elem.tolist()))] = 1
        y_pred_norm.append(line)

    # print(y_pred_norm)
    y_p = np.argmax(np.array(y_pred_norm), 1)
    y_t = np.argmax(np.array(y_test), 1)       

    # print(y_p)
    # print(y_t)
    accu = accuracy_score(y_t, y_p)#, sample_weight=weights)
    wpre = precision_score(y_t, y_p, average='weighted')#, sample_weight=weights)
    mpre = precision_score(y_t, y_p, average='micro')#, sample_weight=weights)
    wrec = recall_score(y_t, y_p, average='weighted')#, sample_weight=weights)
    mrec = recall_score(y_t, y_p, average='micro')#, sample_weight=weights)
    print([accu, wpre, mpre, wrec, mrec])
    return [accu, wpre, mpre, wrec, mrec]

def oversample(df):
    classes = df['label'].value_counts().to_dict()
    most = max(classes.values())
    classes_list = []
    for key in classes:
        classes_list.append(df[df['label'] == key]) 
    classes_sample = []
    for i in range(1,len(classes_list)):
        classes_sample.append(classes_list[i].sample(most, replace=True))
    df_maybe = pd.concat(classes_sample)
    final_df = pd.concat([df_maybe,classes_list[0]], axis=0)
    final_df = final_df.reset_index(drop=True)
    return final_df

if __name__ =="__main__":
    fn = sys.argv[1]
    
    epochs=200
    batch_size=10
    verbose=0

    df = pd.read_csv(fn, sep=',')
    df = df.set_index(df.columns[1])

    data = pd.DataFrame(df, columns = df.columns[1:])
    print(data.columns)
    print(data.shape)
    data = oversample(data)
    print(data.shape)
    
    # data.loc[ df['label'] != 'No_Glitch', 'label'] = "Glitch"
    
    dataset = data.values
    X = dataset[:,0:-1].astype(float)
    Y = dataset[:,-1]

    # print(X.shape[0], X.shape[1])
    # print(Y.shape[0], Y.shape[1])

    scaler_std = StandardScaler()
    X = scaler_std.fit_transform(X)

    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    y = np_utils.to_categorical(encoded_Y)

    y_t = np.argmax(np.array(y), 1)
    unique, counts = np.unique(y_t, return_counts=True)
    class_weights = dict(zip(unique, counts))
    weights = compute_sample_weight(class_weights, y_t)

    no_attributes = X.shape[1]
    no_classes = y.shape[1]

    # For conv1d statement ncols == no_attributes: 
    input_shape = (no_attributes, 1)

    
    input_data = Input(shape=(input_shape), name='Input')

    brench_1 = Conv1D(filters=32, kernel_size=3, input_shape=input_shape, activation='relu')(input_data)
    brench_1 = MaxPooling1D(padding="same")(brench_1)
    brench_1 = Conv1D(filters=64, kernel_size=2, activation='relu')(brench_1)
    brench_1 = MaxPooling1D(padding="same")(brench_1)
    brench_1 = Conv1D(filters=32, kernel_size=1, activation='relu')(brench_1)
    brench_1 = MaxPooling1D(padding="same")(brench_1)
    brench_1 = Conv1D(filters=64, kernel_size=1, activation='relu')(brench_1)
    brench_1 = MaxPooling1D(padding="same")(brench_1)
    brench_1 = Flatten()(brench_1)
    brench_1 = Dense(64, activation='softmax')(brench_1)

    brench_2 = Conv1D(filters=32, kernel_size=3, input_shape=input_shape, activation='relu')(input_data)
    brench_2 = MaxPooling1D(padding="same")(brench_2)
    brench_2 = LSTM(units=32, activation='relu', kernel_initializer='lecun_uniform', return_sequences=True)(brench_2)
    brench_2 = Conv1D(filters=64, kernel_size=2, activation='relu')(brench_2)
    brench_2 = MaxPooling1D(padding="same")(brench_2)
    brench_2 = LSTM(units=64, activation='relu', kernel_initializer='lecun_uniform', return_sequences=True)(brench_2)
    brench_2 = Conv1D(filters=32, kernel_size=1, activation='relu')(brench_2)
    brench_2 = MaxPooling1D(padding="same")(brench_2)
    brench_2 = LSTM(units=32, activation='relu', kernel_initializer='lecun_uniform', return_sequences=True)(brench_2)
    brench_2 = Conv1D(filters=32, kernel_size=1, activation='relu')(brench_2)
    brench_2 = MaxPooling1D(padding="same")(brench_2)
    brench_2 = LSTM(units=32, activation='relu', kernel_initializer='lecun_uniform', return_sequences=True)(brench_2)
    brench_2 = Flatten()(brench_2)
    brench_2 = Dense(64, activation='softmax')(brench_2)

    brench_3 = Conv1D(filters=32, kernel_size=3, input_shape=input_shape, activation='relu')(input_data)
    brench_3 = LSTM(units=32, activation='relu', kernel_initializer='lecun_uniform', return_sequences=True)(brench_3)
    brench_3 = MaxPooling1D(padding="same")(brench_3)
    brench_3 = Conv1D(filters=64, kernel_size=2, activation='relu')(brench_3)
    brench_3 = LSTM(units=64, activation='relu', kernel_initializer='lecun_uniform', return_sequences=True)(brench_3)
    brench_3 = MaxPooling1D(padding="same")(brench_3)
    brench_3 = Conv1D(filters=32, kernel_size=1, activation='relu')(brench_3)
    brench_3 = LSTM(units=32, activation='relu', kernel_initializer='lecun_uniform', return_sequences=True)(brench_3)
    brench_3 = MaxPooling1D(padding="same")(brench_3)
    brench_3 = Conv1D(filters=32, kernel_size=1, activation='relu')(brench_3)
    brench_3 = LSTM(units=32, activation='relu', kernel_initializer='lecun_uniform', return_sequences=True)(brench_3)
    brench_3 = MaxPooling1D(padding="same")(brench_3)
    brench_3 = Flatten()(brench_3)
    brench_3 = Dense(64, activation='softmax')(brench_3)

    brench_4 = LSTM(units=8,  activation='relu',    kernel_initializer='lecun_uniform', return_sequences=True, input_shape=input_shape)(input_data)
    brench_4 = LSTM(units=16, activation='softmax', kernel_initializer='lecun_uniform', return_sequences=True)(brench_4)
    brench_4 = LSTM(units=32, activation='relu',    kernel_initializer='lecun_uniform', return_sequences=True)(brench_4)
    brench_4 = LSTM(units=16, activation='softmax', kernel_initializer='lecun_uniform', return_sequences=True)(brench_4)
    brench_4 = LSTM(units=8,  activation='relu',    kernel_initializer='lecun_uniform', return_sequences=True)(brench_4)
    brench_4 = Flatten()(brench_4)
    brench_4 = Dense(64, activation='softmax')(brench_4)


    combined = Concatenate(name='MODEL_CONCAT')([brench_1, brench_2, brench_3, brench_4])

    output = Dense(units=no_classes, activation = 'softmax', name = 'MODEL_OUTPUT')(combined) #sigmoid #

    model = Model(inputs=input_data, outputs=output, name="DeepWave")
    model.compile(loss ='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

    model.summary()
    crossValidationSSS(model, X, y)
