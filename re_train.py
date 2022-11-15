from pathlib import Path
import json
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from re_chatbot import Chatbot
from retrain_functions import preprocess_dataset, tokenize

def flat_list(list_of_lists):
  return [item for sublist in list_of_lists for item in sublist]

def train():
  INPUT_FOLDER_PATH = Path(__file__).with_name('retrain_data').joinpath('input')
  INTENTS_FILE_PATH = INPUT_FOLDER_PATH.joinpath('intents.json')

  OUTPUT_FOLDER_PATH = Path(__file__).with_name('retrain_data').joinpath('output')
  OUTPUT_FOLDER_PATH.mkdir(parents=True, exist_ok=True)
  TOKENIZER_PATH = OUTPUT_FOLDER_PATH.joinpath('tokenizer.json')
  CONFIG_PATH = OUTPUT_FOLDER_PATH.joinpath('config.json')
  WEIGHTS_PATH = OUTPUT_FOLDER_PATH.joinpath('weights')
  PLOT_PATH = OUTPUT_FOLDER_PATH.joinpath('training.png')

  with open(INTENTS_FILE_PATH, 'r', encoding='utf-8') as rf:
    dataset = json.load(rf)

  questions = flat_list(list(map(lambda x: x['patterns'], dataset)))
  questions_p = preprocess_dataset(questions)

  chatbot = Chatbot()
  input_tensor = tokenize(chatbot.tokenizer, questions_p, None, True)
  with open(TOKENIZER_PATH, 'w') as wf:
    json.dump(chatbot.tokenizer.to_json(), wf)
  
  classes = flat_list(list(map(lambda x: [x['tag']] * len(x['patterns']), dataset)))
  classes_sorted = sorted(set(classes))
  classes_idx = list(map(lambda x: classes_sorted.index(x), classes))

  input_len = len(input_tensor[0])
  output_len = len(set(classes_idx))
  
  with open(CONFIG_PATH, 'w', encoding='utf-8') as wf:
    json.dump({
      "input_len": input_len,
      "output_len": output_len,
      "classes": classes_sorted,
      "intents_to_response": dict(map(lambda x: (x["tag"], x["responses"]), dataset))
    }, wf)
    
  new_input = []
  for idx, sentence in enumerate(input_tensor):
    print(f'Augmenting row {idx} of {len(input_tensor)}', end='\r')
    for _ in range(5):
      new_sentence = [x for x in sentence if x != 0]
      np.random.shuffle(new_sentence)
      new_sentence = new_sentence + [0] * (len(sentence) - len(new_sentence))
      new_sentence = np.array(new_sentence)
      
      new_input.append(new_sentence)
      classes_idx.append(classes_idx[idx])

  input_tensor = list(input_tensor) + new_input
  input_tensor = np.array(input_tensor)

    
  chatbot.build_model(input_len, output_len)
  chatbot.train(input_tensor, classes_idx)
  chatbot.save(WEIGHTS_PATH)
  chatbot.save_plot_accuracy(PLOT_PATH)

if __name__ == '__main__':
  train()