import numpy as np
import re

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation

# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    """
    Convert a list of ordered sequences into smaller window
    :param series: TimeSeries Values
    :param window_size: Length of sequence for modeling
    :return: X, y for training and TimeSeries model
    """
    # containers for input/output pairs
    X = []
    y = []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    """
    Build a simple univariate LSTM Model
    :param window_size: Length of input sequence
    :return: LSTM model
    """
    # Initiate a sequential model
    model = Sequential()
    # Adding LSTM layer
    model.add(LSTM(5, input_shape=(window_size, 1)))
    # Adding a Dense layer -- considering that it is a regression problem, the dimension is 1
    model.add(Dense(1))
    return model

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    """
    :param text: Corpus
    :return: cleaned corpus
    """
    punctuation = ['!', ',', '.', ':', ';', '?']
    # Convert to lower cases
    text = text.lower()
    # Removing any charachter other than a-z and given punctuation
    text = re.sub("[^a-z" + ''.join(punctuation) + " ]", '', text)
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    """
    Convert a list of ordered sequences into smaller window
    :param text: Corpus
    :param window_size: Length of input sequence
    :param step_size: Windows slide size
    :return: X, y for training and TimeSeries model
    """
    # containers for input/output pairs
    inputs = []
    outputs = []
    for i in range(0, len(text) - window_size, step_size):
        inputs.append(text[i:i+window_size])
        outputs.append(text[i+window_size])
    # reshape each
    inputs = np.asarray(inputs)
    inputs.shape = (np.shape(inputs)[0:2])
    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    """
    Build a simple multi-class LSTM Model
    :param window_size: Length of input sequence
    :param num_chars: Number of classes
    :return: LSTM model
    """
    # Initiate a sequential model
    model = Sequential()
    # Adding LSTM layer
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    # Adding a Dense Layer
    model.add(Dense(num_chars))
    # Considering that it is a multi-class problem, Softmax layer is added
    model.add(Activation('softmax'))
    return model
