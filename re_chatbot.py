import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt

class Chatbot:
  def __init__(self, tokenizer_path = None):
    if tokenizer_path is None:
      self.tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<oov>')
    else:
      with open(tokenizer_path, 'r') as rf:
        self.tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json.load(rf))

  def build_model(self, input_len, output_len):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(input_len, input_shape=(input_len,), activation='relu'))
    model.add(tf.keras.layers.Dense(input_len, activation='relu'))
    model.add(tf.keras.layers.Dense(input_len * 2, activation='relu'))
    model.add(tf.keras.layers.Dense(input_len * 4, activation='relu'))
    model.add(tf.keras.layers.Dense(input_len * 8, activation='relu'))
    model.add(tf.keras.layers.Dense(input_len * 16, activation='relu'))
    model.add(tf.keras.layers.Dense(input_len * 32, activation='relu'))
    model.add(tf.keras.layers.Dense(input_len * 16, activation='relu'))
    model.add(tf.keras.layers.Dense(input_len * 8, activation='relu'))
    model.add(tf.keras.layers.Dense(input_len * 4, activation='relu'))
    model.add(tf.keras.layers.Dense(input_len * 2, activation='relu'))
    model.add(tf.keras.layers.Dense(input_len, activation='relu'))
    model.add(tf.keras.layers.Dense(output_len, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    self.nn = model
    
  def train(self, X, Y):
    X = np.array(X)
    Y = tf.one_hot(Y, len(set(Y)))
    
    self.history = self.nn.fit(np.array(X), 
      np.array(Y), 
      epochs=300, 
      batch_size=8,
      verbose=1
    )
    
  def save_plot_accuracy(self, plot_path):
    history = self.history.history['accuracy']
    plt.plot(history)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    
    plt.savefig(plot_path)
    
  def predict(self, question):
    return self.nn.predict(question)
    
  def save(self, path):
    self.nn.save(path)
    
  def load(self, path):
    self.nn = tf.keras.models.load_model(path)