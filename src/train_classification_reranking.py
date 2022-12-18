import random
import shutil

from sentence_transformers import evaluation, LoggingHandler
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader
import torch

from sentence_transformers.cross_encoder import CrossEncoder

import math

import sys
import numpy as np

from myDataset import *
from src.utils import CrossOverGenBleuEvaluator

# Set random seed.
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

def process_data(path, overgen_score_path, greedy_score_path, context_window=3):

    f = open(path)
    dataset = json.load(f)
    f.close()

    f = open(overgen_score_path)
    scores = json.load(f)
    f.close()

    f = open(greedy_score_path)
    greedy_scores = json.load(f)
    f.close()

    assert len(scores) == len(dataset)

    lex_delex_utt_map = load_damd_lex_delex_map()

    data_samples = []

    for entry, score, greedy_score in zip(dataset, scores , greedy_scores):

        gold_delex = entry["resp_text"]
        over_gen_list_delex = entry["over_gen"]
        context_lex = entry["context_text"]
        over_gen_list_delex = over_gen_list_delex

        context_delex = list(map(lambda x: lex_delex_utt_map[x], context_lex))
        sentc = str(" <turn_sep> ".join(context_delex[0 if context_window is None else -context_window:]))
        sentr = gold_delex
        sentn_list = list(map(lambda x: x.replace(" <eos_r>", ""), over_gen_list_delex))
        sentn_list = list(map(lambda x: x.replace(".", " ."), sentn_list))
        sentn_list = list(map(lambda x: x.replace("?", " ?"), sentn_list))

        positive_list = []
        negative_list = []

        for i in range(len(sentn_list)):
            this_label = score[i] >= greedy_score
            if (this_label):
                positive_list.append(InputExample(texts=[sentc, sentn_list[i]], label=this_label))
            else:
                negative_list.append(InputExample(texts=[sentc, sentn_list[i]], label=this_label))
        num_to_added = min(len(positive_list), len(negative_list))

        data_samples.extend(positive_list[:num_to_added])
        data_samples.extend(negative_list[:num_to_added])

    return data_samples



'''
Main function here
'''


def run_experiment():

    batch_size = 64
    epochs = 5
    context_window = 3

    init_model_path = "./output/selection-cross-encoder-quora-distilroberta-base-mintl-ct3-e3-bs64-20_sysonly_delex/"
    note = "0.7_mpnet"

    logger.info("Starting from: " + init_model_path)

    output_path = "./output/classification-mintl-pick-ct"+ str(context_window) +"-e"+str(epochs)+"-bs"+ str(batch_size) +"-seed-" + str(seed) + "-" + note+ "/"
    train_samples = process_data(path="./data/0.7_train.json",
                                 overgen_score_path="./data/sim_score_mpnet_train_0.7.json",
                                 greedy_score_path="./data/sim_score_mpnet_train_0.7_greedy.json",
                                 context_window=context_window)


    assert len(train_samples) > 0

    model = CrossEncoder(init_model_path, num_labels=1)

    model.max_seq_length = 128

    special_tokens = Vocab().special_tokens
    model.tokenizer.add_tokens(special_tokens, special_tokens=True)
    model.model.resize_token_embeddings(len(model.tokenizer))

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)


    evaluator_test = CrossOverGenBleuEvaluator(data_path= "./data/0.7_test.json", context_window = context_window)
    evaluator_dev = CrossOverGenBleuEvaluator(data_path= "./data/0.7_dev.json" , context_window = context_window)

    evaluator = evaluation.SequentialEvaluator([evaluator_test, evaluator_dev], main_score_function=lambda scores: scores[-1])

    warmup_steps = math.ceil(len(train_dataloader) * epochs / batch_size * 0.1)

    model.fit(train_dataloader=train_dataloader,
              evaluator=evaluator,
              epochs=epochs,
              warmup_steps=warmup_steps,
              output_path=output_path)
    shutil.copyfile(sys.argv[0], output_path+"/code.py")


# Main program starts here
def main():
    run_experiment()

## MAIN starts here
if __name__ == '__main__':
    main()