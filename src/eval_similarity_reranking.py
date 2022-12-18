from sentence_transformers import SentenceTransformer, LoggingHandler
import torch
from myDataset import *
from src.utils import KNNOverGenBatchBleuEvaluator

device = 'cuda' if torch.cuda.is_available() else 'cpu'

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)


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

    knn_k = [5000]

    num_of_anchors = [5000]

    anchor_sample_list = []

    for num_anchor in num_of_anchors:
        anchor_samples = process_anchors(path="./data/0.7_train.json",
                                         overgen_score_path="./data/sim_score_mpnet_train_0.7.json",
                                         greedy_score_path="./data/sim_score_mpnet_train_0.7_greedy.json",
                                          context_window=context_window, num_of_anchors=num_anchor)

        assert len(anchor_samples) > 0
        anchor_sample_list.append((num_anchor, anchor_samples))

    init_model_path = "./output/similarity-mintl-ct-3-BatchAllTripletLoss-e5-bs128-seed42-0.7_mpnet/"

    model = SentenceTransformer(init_model_path)

    for anchor_num, anchor_sample in anchor_sample_list:
        evaluator_test = KNNOverGenBatchBleuEvaluator(data_path="./data/0.7_test.json", context_window=context_window,
                                                     anchor_samples=anchor_sample,
                                                     knn_k=knn_k, num_of_anchors=anchor_num)
        evaluator_test.__call__(model)

def main():
    run_experiment()


if __name__ == '__main__':
    main()