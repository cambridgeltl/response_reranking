import json
from sentence_transformers import SentenceTransformer, util

from myDataset.torchDatasets import DummyMinLTOverGenDataset

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

    dataset = DummyMinLTOverGenDataset(file_path)

    print("Size of the dataset:")
    print(len(dataset))

    all_ground_truth = []

    all_preds = []

    for item in dataset:
        all_ground_truth.append(item[0])
        all_preds.append(item[3])

    moedl_path ="sentence-transformers/all-mpnet-base-v2"

    model = SentenceTransformer(moedl_path)

    enc_queries = model.encode(all_ground_truth, convert_to_numpy=True, batch_size=128, normalize_embeddings=True, show_progress_bar=True)

    enc_candidates_list = model.encode(all_preds, convert_to_numpy=True, batch_size=128, normalize_embeddings=True, show_progress_bar=True)

    sim_score_list = []

    assert len(enc_queries) == len(enc_candidates_list)

    for i in zip(enc_queries, enc_candidates_list):
        sim_scores = get_scores(i[0], i[1])
        sim_scores = list(map(lambda x : x.item(), sim_scores))
        sim_scores = roundList(sim_scores)
        sim_score_list.append(sim_scores[0])

    with open(save_path, 'w') as outfile:
        json.dump(sim_score_list, outfile, indent=4)

# Main program starts here
def main():

    run_eval(file_path = "./data/0.7_train.json", save_path = './data/sim_score_mpnet_train_0.7_greedy.json')



## MAIN starts here
if __name__=='__main__':
    main()
