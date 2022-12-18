from itertools import chain

import json
from sentence_transformers import SentenceTransformer, util

from myDataset.torchDatasets import MinLTOverGenDataset
from tqdm import tqdm

'''
Main function here
'''

def roundList(scores):
    return list(map(lambda x : round(x,4), scores))

def get_scores(enc_query, enc_corpus):

    cos_scores = util.pytorch_cos_sim(enc_query, enc_corpus)[0]
    cos_scores = cos_scores.cpu()


    return(cos_scores)

def run_eval(file_path, save_path):

    dataset = MinLTOverGenDataset(file_path)
    # ground_truth_resp, context, candicates

    print("Size of the dataset:")
    print(len(dataset))

    all_ground_truth = []
    all_context = []
    all_candidates = []

    for item in dataset:
        all_ground_truth.append(item[0])
        all_candidates.append(item[2])

    queries, candidates = all_ground_truth, all_candidates

    moedl_path ="sentence-transformers/all-mpnet-base-v2"
    model = SentenceTransformer(moedl_path)

    enc_queries = model.encode(queries, convert_to_numpy=True, batch_size=128, normalize_embeddings=True)
    enc_candidates = []
    num_candicate = len(candidates[0])

    print(num_candicate)
    assert all(map(lambda x :  (len(x) == num_candicate), candidates))
    enc_candidates_list = model.encode(list(chain(*candidates)), convert_to_numpy=True, batch_size=128, normalize_embeddings=True)

    for i in range(0, len(candidates)):
        enc_candidates.append(enc_candidates_list[num_candicate * i: num_candicate * (i + 1)])

    assert len(all_candidates) == len(queries)
    sim_score_list = []
    for i in tqdm(range(len(queries))):
        sim_scores = get_scores(enc_queries[i], enc_candidates[i])
        sim_scores = list(map(lambda x : x.item(), sim_scores))

        sim_score_list.append(sim_scores)

    sim_score_list = list(map(roundList, sim_score_list))

    with open(save_path, 'w') as outfile:
        json.dump(sim_score_list, outfile, indent=4)

# Main program starts here
def main():

    run_eval(file_path = "./data/0.7_train.json", save_path = './data/sim_score_mpnet_train_0.7.json')

## MAIN starts here
if __name__=='__main__':
    main()
