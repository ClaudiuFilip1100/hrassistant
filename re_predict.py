from pathlib import Path
import json
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from re_chatbot import Chatbot
from retrain_functions import preprocess_dataset, tokenize

def predict_answer(question):
  OUTPUT_FOLDER_PATH = Path(__file__).with_name('retrain_data').joinpath('output')

  TOKENIZER_PATH = OUTPUT_FOLDER_PATH.joinpath('tokenizer.json')
  CONFIG_PATH = OUTPUT_FOLDER_PATH.joinpath('config.json')
  WEIGHTS_PATH = OUTPUT_FOLDER_PATH.joinpath('weights')
  
  if not TOKENIZER_PATH.exists():
    raise "No tokenizer found"
  
  if not CONFIG_PATH.exists():
    raise "No config found"
  
  if not WEIGHTS_PATH.exists():
    raise "No config found"
  

  with open(CONFIG_PATH, 'r', encoding='utf-8') as rf:
    config = json.load(rf)

  input_len = config["input_len"]
  output_len = config["output_len"]
  classes = config["classes"]
  intents_to_response = config["intents_to_response"]
  
  chatbot = Chatbot(TOKENIZER_PATH)
  chatbot.load(WEIGHTS_PATH)
  question_p = preprocess_dataset([question])
  input_tensor = tokenize(chatbot.tokenizer, question_p, input_len, False)
  
  res = chatbot.predict(input_tensor)[0]
  pred = np.argmax(res)
  class_name = classes[pred]
  return {
    "answer": np.random.choice(intents_to_response[class_name]) if class_name in intents_to_response.keys() and res[pred] > 0.7 else 'Nu sunt sigur ca pot raspunde la aceasta intrebare',
    "confidence": res[pred]
  }

if __name__ == '__main__':
  q = input('Your question: ')
  
  while q != 'quit':
    print(f"Q: {q}")
    print(f"A: {predict_answer(q)}")

    q = input('Your question: ')
