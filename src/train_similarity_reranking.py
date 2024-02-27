import shutil

from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation, \
    LoggingHandler
from torch.utils.data import DataLoader
import torch
import random


import math

import sys
import numpy as np


from myDataset import *
from src.utils import KNNOverGenBatchBleuEvaluator

device = 'cuda' if torch.cuda.is_available() else 'cpu'

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

import torch

# Set random seed.
seed = 42

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


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
    assert len(dataset[0]["over_gen"]) == len(scores[0])

    lex_delex_utt_map = load_damd_lex_delex_map()

    data_samples = []

    for entry, score, greedy_score in zip(dataset[:], scores[:],
                                        greedy_scores[:]):

        gold_delex = entry["resp_text"]
        over_gen_list_delex = entry["over_gen"]
        context_lex = entry["context_text"]

        context_delex = list(map(lambda x: lex_delex_utt_map[x], context_lex))
        sentc = str(" <turn_sep> ".join(context_delex[0 if context_window is None else -context_window:]))
        sentr = gold_delex
        sentn_list = list(map(lambda x: x.replace(" <eos_r>", ""), over_gen_list_delex))

        positive_list = []
        negative_list = []

        for i in range(len(sentn_list)):
            if (score[i] >= greedy_score):
                positive_list.append(InputExample(texts=[sentc + " <turn_sep> " + sentn_list[i]], label=1))
            else:
                negative_list.append(InputExample(texts=[sentc + " <turn_sep> " + sentn_list[i]], label=0))
        num_to_added = min(len(positive_list), len(negative_list))

        data_samples.extend(positive_list[:num_to_added])
        data_samples.extend(negative_list[:num_to_added])

    return data_samples


def process_anchors(path, overgen_score_path, greedy_score_path, context_window=3, num_of_anchors=1000):
    f = open(path)
    dataset = json.load(f)
    f.close()

    f = open(overgen_score_path)
    scores = json.load(f)
    f.close()

    greedy_score_path = greedy_score_path
    f = open(greedy_score_path)
    greedy_scores = json.load(f)
    f.close()

    assert len(scores) == len(dataset)
    assert len(dataset[0]["over_gen"]) == len(scores[0])

    assert len(dataset) > num_of_anchors

    lex_delex_utt_map = load_damd_lex_delex_map()

    anchor_samples = []

    for entry, score, greedy_score in zip(dataset[-num_of_anchors:], scores[-num_of_anchors:],
                                        greedy_scores[-num_of_anchors:]):

        gold_delex = entry["resp_text"]
        over_gen_list_delex = entry["over_gen"]
        context_lex = entry["context_text"]

        context_delex = list(map(lambda x: lex_delex_utt_map[x], context_lex))
        sentc = str(" <turn_sep> ".join(context_delex[0 if context_window is None else -context_window:]))
        sentr = gold_delex
        sentn_list = list(map(lambda x: x.replace(" <eos_r>", ""), over_gen_list_delex))

        positive_list = []
        negative_list = []

        for i in range(len(sentn_list)):
            if (score[i] >= greedy_score):
                positive_list.append((sentc + " <turn_sep> " + sentn_list[i], 1))
            else:
                negative_list.append((sentc + " <turn_sep> " + sentn_list[i], 0))
        num_to_added = min(len(positive_list), len(negative_list))

        anchor_samples.extend(positive_list[:num_to_added])
        anchor_samples.extend(negative_list[:num_to_added])

    return anchor_samples


'''
Main function here
'''


def run_experiment():

    context_window = 3

    knn_k = [500, 1000, 5000]

    num_of_anchors = [5000, 10000]

    anchor_sample_list = []

    batch_size = 128

    train_samples = process_data(path="./data/0.7_train.json",
                                 overgen_score_path="./data/sim_score_mpnet_train_0.7.json",
                                 greedy_score_path="./data/sim_score_mpnet_train_0.7_greedy.json",
                                 context_window=context_window)


    assert len(train_samples) > 0

    for num_anchor in num_of_anchors:
        anchor_samples = process_anchors(path="./data/0.7_train.json",
                                         overgen_score_path="./data/sim_score_mpnet_train_0.7.json",
                                         greedy_score_path="./data/sim_score_mpnet_train_0.7_greedy.json",
                                          context_window=context_window, num_of_anchors=num_anchor)

        assert len(anchor_samples) > 0
        anchor_sample_list.append((num_anchor, anchor_samples))

    init_model_path = "./output/selection_google-bert-bert-base-uncased_delex/"

    epochs = 5
    note = "0.7_mpnet"

    model = SentenceTransformer(init_model_path)
    model.max_seq_length = 128

    # Adding the special tokens.
    word_embedding_model = model._first_module()
    special_tokens = Vocab().special_tokens
    word_embedding_model.tokenizer.add_tokens(special_tokens, special_tokens=True)
    word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))

    train_loss = losses.BatchAllTripletLoss(model=model)

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)

    all_evaluator_list = []
    for anchor_num, anchor_sample in anchor_sample_list:
        evaluator_test = KNNOverGenBatchBleuEvaluator(data_path="./data/0.7_test.json", context_window=context_window,
                                                     anchor_samples=anchor_sample,
                                                     knn_k=knn_k, num_of_anchors=anchor_num)

        all_evaluator_list.append(evaluator_test)
        evaluator_dev = KNNOverGenBatchBleuEvaluator(data_path="./data/0.7_dev.json", context_window=context_window,
                                                     anchor_samples=anchor_sample,
                                                     knn_k=knn_k, num_of_anchors=anchor_num)

        all_evaluator_list.append(evaluator_dev)


    evaluator = evaluation.SequentialEvaluator(all_evaluator_list, main_score_function=lambda scores: np.max(scores))

    warmup_steps = math.ceil(len(train_dataloader) * epochs / batch_size * 0.1)

    loss_str = train_loss.__str__()
    loss_str = loss_str[:loss_str.index("(")]
    output_path = "./output/reranking_similarity_" + note + "/"

    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epochs, evaluator=evaluator,
              warmup_steps=warmup_steps, output_path=output_path)

    shutil.copyfile(sys.argv[0], output_path + "/code.py")


def main():
    run_experiment()


if __name__ == '__main__':
    main()