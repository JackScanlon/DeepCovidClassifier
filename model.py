import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Activation, BatchNormalization, Dense, Flatten, Dropout, TimeDistributed, LSTM, Input
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, recall_score, precision_score, accuracy_score

class DeepCovidClassifier():
  def __init__(self, processor, dir='./models', name='covidCNN.h5'):
    self.processor = processor
    self.channels = processor.channels
    self.classes = processor.classes
    self.width = processor.imsize[0]
    self.height = processor.imsize[1]
    self.name = name
    self.model_dir = dir

    return

  def create(self, summary=False, weights=None, training=False):
    model = Sequential([
      Conv2D(32, (3, 3), input_shape=(self.channels, self.height, self.width), data_format='channels_first', activation='relu', name='conv1'),
      Conv2D(32, (3, 3), activation='relu', name='conv2'),
      Activation('relu'),
      BatchNormalization(name='norm'),
      MaxPooling2D(pool_size=(2, 2), name='pool1'),
      Conv2D(64, (3, 3), activation='relu', name='conv3'),
      MaxPooling2D(pool_size=(2, 2), name='pool2'),
      Conv2D(128, (3, 3), activation='relu', name='conv4'),
      MaxPooling2D(pool_size=(2, 2), name='pool3'),
      Dropout(0.2, name='drop'),
      Flatten(),
      Dense(128, activation='relu', name='dense'),
      Dense(self.classes, activation='sigmoid', name='output')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    if weights:
      model.load_weights(weights, by_name=True)
    
    if not training:
      for l in model.layers:
        l.trainable = False

    if summary:
      print(model.summary())

    return model

  def optimise(self, params, refit_score):
    model = KerasClassifier(build_fn=self.create, verbose=1)
    grid = GridSearchCV(
      estimator=model,
      param_grid=params,
      cv=StratifiedKFold(n_splits=5, random_state=None),
      refit=refit_score,
      scoring={
        'recall_score': make_scorer(recall_score),
        'precision_score': make_scorer(precision_score),
        'accuracy_score': make_scorer(accuracy_score)
      },
      n_jobs=-1
    )

    x_train, y_train = self.processor.get_train_data()
    g_res = grid.fit(x_train, y_train)
    print(f"Score {g_res.best_score_:.4f} with params {g_res.best_params_} using score {refit_score}")
    print(pd.DataFrame(g_res.cv_results_))

  def assess(self, epochs=64, batch_size=256, verbose=1):
    strat = StratifiedKFold(n_splits=5, shuffle=True)

    score = []; history = []; folds = 0

    x, y = self.processor.get_train_data()
    for train, test in strat.split(x, y):
      folds += 1
      print(f"Fold {folds}")

      model = self.create()
      hx = model.fit(x[train], y[train], epochs=epochs, batch_size=batch_size, validation_data=(x[test], y[test]), verbose=verbose).history
      sc = model.evaluate(x[test], y[test], verbose=verbose)
      score.append(sc)
      history.append(hx)
    
    print(f"Average test loss of {np.asarray(score)[:,0].mean():.4f} and average test accuracy of {np.asarray(score)[:,1].mean():.4f}")

    return score, history

  def train(self, epochs=64, batch_size=256, verbose=1, summary=False):
    model = self.create(summary=summary, training=True)
    
    x_train, y_train = self.processor.get_train_data()
    stop = EarlyStopping(monitor='val_loss', patience=15, mode='min')
    save = ModelCheckpoint(f"{self.model_dir}/{self.name}", save_best_only=True, monitor='val_loss', mode='min', save_weights_only=True)
    history = model.fit(x_train, y_train, validation_split=0.2, epochs=epochs, verbose=verbose, batch_size=batch_size, callbacks=[stop, save])

    return model, history