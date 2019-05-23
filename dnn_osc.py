from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn import decomposition, preprocessing
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split

import tensorflow as tf
import keras as K
import tensorflow as tf
from keras import backend
from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, History
from keras.utils import np_utils

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import os, csv, time, shutil
import cv_data, settings
import ast
import gc
import seaborn as sns

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#plots train and validation errors as a function of training steps
def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[0].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('MSE')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylim(0,1000)
    axs[0].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    plt.show()


def averageOverLastN(a, n=3):
    return np.mean(a[-n:])

def main():
  section = 'Settings'
  args = dict.fromkeys(settings.args_keys)

  #read hyperparameters from Settings.ini
  for setting in settings.args_keys:
    value = settings.get_setting(settings.path, section, setting)
    args[setting] = ast.literal_eval(value)
    
  print(args)

  tf.reset_default_graph()
  start_time = time.time()
  

  #initialize statistics to be collected
  mse_test = []
  mae_test = []
  mae_std_test =[]
  preds = []
  r2_test = []
  val_average_over_last_n = []
  tr_average_over_last_n = []

  
  (XXX, yyy) = cv_data.load_data()
  XXt, X, yyt, y = train_test_split(XXX, yyy, test_size=args['train_percent'], random_state=42)

  #hartree to meV conversion of y values
  y *= args['reorg_norm_factor']
  yyt *= args['reorg_norm_factor']

  cv_split = []
  
  #for stratified split
  if(args['stratified']):
    min = 0
    max = len(X)
    for i in range(args['cv']):
      test_ind=np.arange(min,max,args['cv'])
      train_ind = np.arange(0,max)
      mask = np.ones(max,dtype=bool)
      mask[test_ind] = False
      train_ind = train_ind[mask]
      cv_split.append([train_ind,test_ind])
      min += 1
    print('*** Done stratification ***') 
  else:
    kf = KFold(n_splits=args['cv'], shuffle=True, random_state=None)
    cv_split=kf.split(X)
    print('*** Done K-Fold ***') 
  
  #k-fold cross validation loop
  foldNumber = 1
  for train_index, test_index in cv_split:
    start_time_fold = time.time()
    print('\n --- %s. Fold started --- ' % (foldNumber))
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    

    if args['normalization']:
      max_abs_scaler = preprocessing.MaxAbsScaler()
      X_train_maxabs = max_abs_scaler.fit_transform(X_train.values)
      X_test_maxabs = max_abs_scaler.transform(X_test.values)
      X_train2 = pd.DataFrame(X_train_maxabs, columns=X_train.columns, index=X_train.index)
      X_test2 = pd.DataFrame(X_test_maxabs, columns=X_test.columns, index=X_test.index)      
      X_train = X_train2
      X_test = X_test2
      print('*** Done normalizing ***') 

    if args['pca']:
      pca = decomposition.PCA(n_components=args['pcaVec'])
      pca.fit(X_train)
      columns = ['pca_%i' % i for i in range(args['pcaVec'])]
      X_train2 = pd.DataFrame(pca.transform(X_train), columns=columns, index=X_train.index)
      X_test2 = pd.DataFrame(pca.transform(X_test), columns=columns, index=X_test.index)
      X_train = X_train2
      X_test = X_test2
      print('*** Done PCA ***') 
    
  

    config = tf.ConfigProto(
         allow_soft_placement=True,
         gpu_options = tf.GPUOptions(allow_growth=True))
    set_session(tf.Session(config=config))

    print("X_train shape:", X_train.shape[1])

    #build the neural network
    model = Sequential()
    layers=args['hidden_units']
    for i in range(len(layers)):
      if i==0: #first hidden layer
        model.add(Dense(layers[i], input_shape=(X_train.shape[1],), activation='relu', 
                        kernel_initializer='he_normal'))
        model.add(Dropout(args['dropout']))
      elif i==len(layers)-1: #output layer
        model.add(Dense(layers[i], activation='linear', kernel_initializer="he_normal"))
      else:
        model.add(Dense(layers[i], activation='relu', kernel_initializer="he_normal"))
        model.add(Dropout(args['dropout']))

    model.compile(loss='mean_squared_error', optimizer=K.optimizers.Adam(lr=args['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0, amsgrad=False))
    #model.compile(loss='mean_squared_error', optimizer=K.optimizers.RMSprop(lr=args['learning_rate'], rho=0.9,  epsilon=None, decay=0.0))
    #model.compile(loss='mean_squared_error', optimizer= K.optimizers.SGD(lr=args['learning_rate'], momentum=0.9, decay=0.0, nesterov=False))
    
    history = History()
    
    hist = model.fit(X_train, y_train, epochs=args['train_steps'], verbose=1, shuffle=True, batch_size=args['batch_size'], validation_data=(X_test, y_test))
    val_loss = hist.history['val_loss']
    loss = hist.history['loss']
    
   
    #average over last 20 validation loss values
    average_over = 20
    mov_av_val = averageOverLastN(np.sqrt(np.array(val_loss)), average_over) # moving average of RMSE
    val_average_over_last_n.append(mov_av_val)
    
    #average over last 20 train loss values
    mov_av_tr = averageOverLastN(np.sqrt(np.array(loss)), average_over)
    tr_average_over_last_n.append(mov_av_tr)
    
  
    print('*** Done training ***')    

    # Evaluate how the model performs on data it has not yet seen.

    y_hat=model.predict(X_test, batch_size=args['batch_size'])
    y_train_hat=model.predict(X_train, batch_size=args['batch_size'])
    print('*** Done predictions ***')
    
    predictions = list(p[0] for p in y_hat) 
    pred_train = list(p[0] for p in y_train_hat) 

    error = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mae_std = np.std(np.abs(y_test-predictions))
    r2 = r2_score(y_test, predictions)
    
    mse_test.append(error)
    r2_test.append(r2)
    mae_test.append(mae)
    mae_std_test.append(mae_std)
    

    print('\nTest MSE: ', error)
    print('\nR2 Score: ', r2)
    
    hrs,mins,sec = calculate_time(start_time_fold,time.time())
    print('\n --- %s. Fold Time: %s h, %s min %s sec ---' % (foldNumber,hrs,mins,sec))

    model.reset_states()
    K.backend.clear_session()
    gc.collect()
    
    foldNumber += 1
  #endcvfor

  

  #average over cv folds
  cv_last_val_rmse = np.mean(np.sqrt(mse_test))
  cv_last_val_std_rmse = np.std(np.sqrt(mse_test))
  cv_last_val_mae = np.mean(mae_test)
  cv_last_val_std_mae = np.std(mae_test)
  cv_average_tr_rloss = np.mean(tr_average_over_last_n)
  cv_r2 = sum(r2_test)/float(len(r2_test)) 
  cv_average_val_rloss = np.mean(val_average_over_last_n)
  

  print('all the mse_test: ',mse_test)
  print('all the rmse_test: ', np.sqrt(mse_test))
  print('all the mae_test: ',mae_test)
  print('all the mae_std_test', mae_std_test)
  print('all the r2_test',r2_test)
  print('\n Average RMSE Train Loss :', cv_average_tr_rloss)
  print('Average R2 on Test:',cv_r2)
  print('\n Average Validation RMSE Loss on Test:', cv_average_val_rloss)
  

  
  hrs,mins,sec = calculate_time(start_time,time.time())  
  print('\n --- Total Time: %s h, %s min %s sec ---' % (hrs,mins,sec))


  #print statistics to csv file
  header = ['Last_R2','Last_Val_RMSE', 'Last_Val_Std_RMSE',  'Last_Val_MAE', 'Last_Val_Std_MAE', 'Average_Val_RMSE', 'Average_TR_RMSE','DataSize']
  fields = [float(cv_r2),  float(cv_last_val_rmse), float(cv_last_val_std_rmse),  float(cv_last_val_mae), float(cv_last_val_std_mae), float(cv_average_val_rloss),  float(cv_average_tr_rloss), len(X)]
  for key in settings.args_keys: 
    header.append(key)
    fields.append(settings.get_setting(settings.path,section,key))

  
  filename = settings.get_setting(settings.path,section,'csv_output_file')
  file_exists = os.path.isfile(filename)
  with open(filename, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    if not file_exists:
      writer.writerow(header);
    writer.writerow(fields);
    print('*** Done writing csv ***')    

# Calculate elapsed time
def calculate_time(s,f):
    sec = f - s
    hrs = int(sec / 3600)
    sec -= 3600*hrs
    mins = int(sec / 60)
    sec -= 60*mins
    return hrs,mins,sec
  
def savefig(y, preds, fields='None', show=False):
  plt.plot(y, preds, '.')
  if(show):
    plt.show()
  plt.savefig('{0}.png'.format(fields))
  print('*** Done saving plt ***')    

def from_dataset(ds):
    return lambda: ds.make_one_shot_iterator().get_next()

