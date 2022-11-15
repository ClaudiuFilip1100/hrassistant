
# Functions
import re 
import tensorflow as tf

def remove_preposition(text):
  q_regex = 'Intrebarea\ [0-9]+\:\ '
  a_regex = 'Raspuns\ [0-9]+\:\ '
  
  if 'Intrebarea' in text:
    text = re.sub(q_regex, '', text)
  else:
    text = re.sub(a_regex, '', text)
  return text

# remove punctuation
import string

def remove_punctuation(text):
  punctuationfree="".join([i for i in text if i not in string.punctuation])
  return punctuationfree

#remove stopwords
def remove_stopwords(text, path_to_stopwords):
  stopwords = open(path_to_stopwords).readline().split(',')
  return [x for x in text if x not in stopwords]

# all lowercase
def to_lower(text):
  return text.lower()

def update_romanian_marks(text):
  romanian = [['ă', 'a'], ['â', 'a'], ['î', 'i'], ['ș', 's'], ['ț', 't']]
  for ro in romanian:
    text = text.replace(ro[0], ro[1])
  
  return text

# split into words
def split_words(text):
  tokens = re.split(' ',text)
  return tokens

# stemming
from nltk.stem.snowball import RomanianStemmer

def stemming(text):
  snow_stemmer = RomanianStemmer()
  return [snow_stemmer.stem(word) for word in text]

def preprocess_sentence(sentence):
  text = remove_preposition(sentence)
  text = remove_punctuation(text)
  text = update_romanian_marks(text)
  text = split_words(text)
  # text = remove_stopwords(text, 'data/input/stopwords-ro.txt')
  text = list(filter(None, text))
  text = [to_lower(x) for x in text]
  text = stemming(text)
  text = ['<start>'] + text 
  text = text + ['<end>']
  
  return text 
    
def preprocess_dataset(_list):
  return [preprocess_sentence(text) for text in _list]

def tokenize(tokenizer, lang, pad_len = None, fit = False):
  # lang = list of sentences in a language

  # print(len(lang), "example sentence: {}".format(lang[0]))
  # lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<oov>')
  if fit:
    tokenizer.fit_on_texts(lang)

  ## tf.keras.preprocessing.text.Tokenizer.texts_to_sequences converts string (w1, w2, w3, ......, wn) 
  ## to a list of correspoding integer ids of words (id_w1, id_w2, id_w3, ...., id_wn)
  tensor = tokenizer.texts_to_sequences(lang) 

  ## tf.keras.preprocessing.sequence.pad_sequences takes argument a list of integer id sequences 
  ## and pads the sequences to match the longest sequences in the given input
  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post', maxlen=pad_len)

  return tensor