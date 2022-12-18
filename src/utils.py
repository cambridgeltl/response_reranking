from itertools import chain

import torch
from sentence_transformers.evaluation import SentenceEvaluator
import numpy as np
import logging
import os

import json
from sentence_transformers import util, SentenceTransformer

from myDataset import MinLTOverGenDataset, prepare_woz22_data_delex, prepare_woz22_data
from myMetrics import *
from sklearn.neighbors import KNeighborsRegressor
logger = logging.getLogger(__name__)


def get_gold_idx(corpus, correct):
    for i in range(len(corpus)):
        if corpus[i] == correct:
            return i
    return -1

def get_best_idx(enc_query, enc_corpus, score_fn = "cos"):

    if score_fn == "cos":
        cos_scores = util.pytorch_cos_sim(enc_query, enc_corpus)[0]
        cos_scores = cos_scores.cpu()
        top_results = torch.topk(cos_scores, k=10)
        best_idx = int(top_results[1][0])
    elif score_fn == "dot":
        dot_scores = (enc_query * enc_corpus).sum(-1)
        best_idx =np.argmax(dot_scores)
    else:
        raise NotImplementedError()

    return(best_idx)

class CrossOverGenBleuEvaluator(SentenceEvaluator):

    def __init__(self, data_path, context_window=3, candidate_size = None):

        self.data_path = data_path
        self.eval_dataset = MinLTOverGenDataset(self.data_path, context_window=context_window)
        self.candidate_size = candidate_size
        logger.info("Size of the dataset:\t{}".format(len(self.eval_dataset)))
        logger.info("Random baseline on "+  self.data_path+ ":")

        metrics = [
            METEOR(),
            ROUGE(),
            CorpusBLEU(),
        ]

        for item in self.eval_dataset:
            ground_truth = item[0]
            generated_output = item[2][0]
            for metric in metrics:
                metric.update((generated_output, ground_truth))

        for metric in metrics:
            name = metric.name()
            score = metric.compute()
            logger.info("{:s} = {:s}".format(name, str(score)))

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1, prediction_output_path: str = None) -> float:

        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info(self.data_path + " evaluation" + out_txt)

        all_ground_truth = []
        all_context = []
        all_candidates = []

        if self.candidate_size != None:
            for item in self.eval_dataset:
                all_ground_truth.append(item[0])
                all_context.append(item[1])
                all_candidates.append(item[2][:self.candidate_size])
        else:
            for item in self.eval_dataset:
                all_ground_truth.append(item[0])
                all_context.append(item[1])
                all_candidates.append(item[2])

        metrics = [
            # BLEU(),
            # MultiWoZBLEU(),
            METEOR(),
            ROUGE(),
            CorpusBLEU(),
        ]

        total = 0
        reranked_output = []
        output_eval_list = []

        query_num = len(all_context)
        query_len = len(all_candidates[0])

        all_query_list = []

        for i in range(len(all_context)):
            this_query = all_context[i]
            this_candidate = all_candidates[i]
            this_answer = all_ground_truth[i]

            total += 1

            queries_candidate_pair_list = list(
                map(lambda x: [x[0], x[1]], zip([this_query for _ in this_candidate], this_candidate)))
            all_query_list.append(queries_candidate_pair_list)

        temp_score_list = model.predict(list(chain(*all_query_list)))


        scores_list = []
        all_index_list = []
        for i in range(query_num):
            scores_list.append(temp_score_list[query_len * i: query_len * (i + 1)])

        assert len(scores_list) == len(all_context)
        assert len(scores_list[0]) == len(all_candidates[0])

        for i in range(len(scores_list)):
            scores = scores_list[i]
            this_query = all_context[i]
            this_candidate = all_candidates[i]
            this_answer = all_ground_truth[i]

            temp_eval = {}
            temp_eval["context"] = this_query
            temp_eval["ans"] = this_answer
            temp_eval["out"] = list(map(lambda x: (x[0], "{:.2f}".format(x[1])),
                                        sorted(list(zip(this_candidate, scores)), reverse=True,
                                               key=lambda tup: tup[1])))

            output_eval_list.append(temp_eval)
            best_idx = np.argmax(scores)
            reranked_output.append(this_candidate[best_idx])
            all_index_list.append(int(best_idx))
            for metric in metrics:
                metric.update((this_candidate[best_idx], all_ground_truth[i]))

        all_eval_scores = []
        for metric in metrics:
            name = metric.name()
            score = metric.compute()
            all_eval_scores.append(score)
            logger.info("{:s} = {:s}".format(name, str(score)))

        if output_path is not None:

            if "test" in self.data_path:
                pre_fix = "test_"
            elif "dev" in self.data_path or "val" in self.data_path:
                pre_fix = "dev_"
            else:
                pre_fix = ""

            json_path = os.path.join(output_path, pre_fix + "output_scores.json")
            output_file_exists = os.path.isfile(json_path)
            with open(json_path, newline='', mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                json.dump(all_eval_scores, f, ensure_ascii=False, indent=4)



        if prediction_output_path is not None:
            temp_putput_dic = {}
            temp_putput_dic["output"] = reranked_output
            temp_putput_dic["index"] = all_index_list

            with open(prediction_output_path, newline='', mode='w', encoding="utf-8") as f:
                json.dump(temp_putput_dic, f, ensure_ascii=False, indent=4)

        return all_eval_scores[-1]

class BiOverGenKNNBleuEvaluator(SentenceEvaluator):

    def __init__(self, anchor_samples, num_of_anchors, data_path, context_window=3, sim_fn = "cos",
                 knn_k = 3, batch_size = 256, candidate_size = None):

        self.data_path = data_path
        self.eval_dataset = MinLTOverGenDataset(self.data_path, context_window=context_window)
        self.anchor_samples = anchor_samples
        self.knn_k = knn_k
        self.num_of_anchors = num_of_anchors
        self.sim_fn = sim_fn
        self.encoding_bs = batch_size

        self.candidate_size = candidate_size

        logger.info("Size of the dataset:\t{}".format(len(self.eval_dataset)))
        logger.info("Random baseline on "+  self.data_path+ ":")


        metrics = [
            METEOR(),
            ROUGE(),
            CorpusBLEU(),
        ]

        for item in self.eval_dataset:
            ground_truth = item[0]
            generated_output = item[2][0]
            for metric in metrics:
                metric.update((generated_output, ground_truth))

        for metric in metrics:
            name = metric.name()
            score = metric.compute()
            logger.info("{:s} = {:s}".format(name, str(score)))



    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1, prediction_output_path: str = None) -> float:


        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info(self.data_path + " evaluation with k="  +  str(self.knn_k)   + " num_of_anchors=" +  str(self.num_of_anchors) + out_txt)


        all_composed_query = []
        knn_archor_labels = []
        for query, label in self.anchor_samples:
            all_composed_query.append(query)
            knn_archor_labels.append(label)

        knn_archor_embeddings = model.encode(all_composed_query, convert_to_numpy=True, batch_size = self.encoding_bs,
                                           normalize_embeddings=True, show_progress_bar=False)


        assert len(knn_archor_embeddings) == len(knn_archor_labels)

        neigh = KNeighborsRegressor(n_neighbors=self.knn_k)
        neigh.fit(knn_archor_embeddings, knn_archor_labels)


        # Encoding testing instances

        all_ground_truth = []
        all_context = []
        all_candidates = []


        if self.candidate_size != None:
            for item in self.eval_dataset:
                all_ground_truth.append(item[0])
                all_context.append(item[1])
                all_candidates.append(item[2][:self.candidate_size])
        else:
            for item in self.eval_dataset:
                all_ground_truth.append(item[0])
                all_context.append(item[1])
                all_candidates.append(item[2])



        queries, candidates = all_context, all_candidates

        all_composed_query = []
        for query, candidate_list in zip(queries,candidates):
            composed_query = list(map(lambda x : query + " <turn_sep> " + x,   candidate_list))
            all_composed_query.extend(composed_query)

        enc_candidates_list = model.encode(all_composed_query, convert_to_numpy=True, batch_size = self.encoding_bs,
                                           normalize_embeddings=True)
        all_prediction_list = neigh.predict(enc_candidates_list)

        enc_candidates = []
        num_candicate = len(candidates[0])

        for i in range(0, len(candidates)):
            enc_candidates.append(all_prediction_list[num_candicate * i: num_candicate * (i + 1)])




        metrics = [
            METEOR(),
            ROUGE(),
            CorpusBLEU(),
        ]
        all_index_list = []
        reranked_output = []
        assert len(all_candidates) == len(queries)
        for i in range(len(queries)):



            best_idx = np.argmax(enc_candidates[i])
            reranked_output.append(candidates[i][best_idx])
            all_index_list.append(int(best_idx))

            for metric in metrics:
                metric.update((candidates[i][best_idx], all_ground_truth[i]))

        all_eval_scores = []
        for metric in metrics:
            name = metric.name()
            score = metric.compute()
            all_eval_scores.append(score)
            logger.info("{:s} = {:s}".format(name, str(score)))

        if output_path is not None:

            if "test" in self.data_path:
                pre_fix = "test_"
            elif "dev" in self.data_path or "val" in self.data_path:
                pre_fix = "dev_"
            else:
                pre_fix = ""

            json_path = os.path.join(output_path, pre_fix + "output_scores.json")
            output_file_exists = os.path.isfile(json_path)
            with open(json_path, newline='', mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                json.dump(all_eval_scores, f, ensure_ascii=False, indent=4)


        if prediction_output_path is not None:
            temp_putput_dic = {}
            temp_putput_dic["output"] = reranked_output
            temp_putput_dic["index"] = all_index_list

            with open(prediction_output_path, newline='', mode='w', encoding="utf-8") as f:
                json.dump(temp_putput_dic, f, ensure_ascii=False, indent=4)


        return all_eval_scores[-1]

class KNNOverGenBatchBleuEvaluator(SentenceEvaluator):

    def __init__(self, anchor_samples, num_of_anchors, data_path , context_window=3, sim_fn = "cos",
                 knn_k=None, batch_size = 256):

        if knn_k is None:
            knn_k = [1000]


        self.data_path = data_path
        self.eval_dataset = MinLTOverGenDataset(self.data_path, context_window=context_window)
        self.anchor_samples = anchor_samples
        self.knn_k_list = knn_k
        self.num_of_anchors = num_of_anchors
        self.sim_fn = sim_fn
        self.encoding_bs = batch_size
        self.best_output = None

        logger.info("Size of the dataset:\t{}".format(len(self.eval_dataset)))
        logger.info("Random baseline on "+  self.data_path+ ":")


        metrics = [
            METEOR(),
            ROUGE(),
            CorpusBLEU(),
        ]

        for item in self.eval_dataset:
            ground_truth = item[0]
            generated_output = item[2][0]
            for metric in metrics:
                metric.update((generated_output, ground_truth))

        for metric in metrics:
            name = metric.name()
            score = metric.compute()
            logger.info("{:s} = {:s}".format(name, str(score)))

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:


        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info(self.data_path + " evaluation with num_of_anchors=" +  str(self.num_of_anchors) + out_txt)


        all_composed_query = []
        knn_archor_labels = []
        for query, label in self.anchor_samples:
            all_composed_query.append(query)
            knn_archor_labels.append(label)

        knn_archor_embeddings = model.encode(all_composed_query, convert_to_numpy=True, batch_size = self.encoding_bs,
                                           normalize_embeddings=True, show_progress_bar=True)


        assert len(knn_archor_embeddings) == len(knn_archor_labels)

        knn_regressor_list = []
        for temp_k in self.knn_k_list:
            neigh = KNeighborsRegressor(n_neighbors=temp_k)
            neigh.fit(knn_archor_embeddings, knn_archor_labels)
            knn_regressor_list.append(neigh)



        all_ground_truth = []
        all_context = []
        all_candidates = []
        for item in self.eval_dataset:
            all_ground_truth.append(item[0])
            all_context.append(item[1])
            all_candidates.append(item[2][:20])



        queries, candidates = all_context, all_candidates

        all_composed_query = []
        for query, candidate_list in zip(queries,candidates):
            composed_query = list(map(lambda x : query + " <turn_sep> " + x,   candidate_list))
            all_composed_query.extend(composed_query)

        enc_candidates_list = model.encode(all_composed_query, convert_to_numpy=True, batch_size = self.encoding_bs,
                                           normalize_embeddings=True)


        output_eval_dic = {}
        for neigh in knn_regressor_list:
            logger.info("KNN with k= {:s}".format(str(neigh.n_neighbors)))

            all_prediction_list = neigh.predict(enc_candidates_list)

            enc_candidates = []
            num_candicate = len(candidates[0])

            for i in range(0, len(candidates)):
                enc_candidates.append(all_prediction_list[num_candicate * i: num_candicate * (i + 1)])
            assert len(all_candidates) == len(queries)

            metrics = [
                METEOR(),
                ROUGE(),
                CorpusBLEU(),
            ]

            reranked_output = []
            for i in range(len(queries)):
                best_idx = np.argmax(enc_candidates[i])
                reranked_output.append(candidates[i][best_idx])

                for metric in metrics:
                    metric.update((candidates[i][best_idx], all_ground_truth[i]))

            all_eval_scores = []
            for metric in metrics:
                name = metric.name()
                score = metric.compute()
                all_eval_scores.append(score)
                logger.info("{:s} = {:s}".format(name, str(score)))
            output_eval_dic[neigh.n_neighbors] = all_eval_scores


        if output_path is not None:

            if "test" in self.data_path:
                pre_fix = "test_"
            elif "dev" in self.data_path or "val" in self.data_path:
                pre_fix = "dev_"
            else:
                pre_fix = ""

            json_path = os.path.join(output_path, pre_fix + "output_scores.json")
            output_file_exists = os.path.isfile(json_path)
            with open(json_path, newline='', mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                json.dump(output_eval_dic, f, ensure_ascii=False, indent=4)

        temp_all_bleu_list = []
        for knn_k in output_eval_dic:
            eval_scores = output_eval_dic[knn_k]
            temp_all_bleu_list.append(eval_scores[-1])

        best_score = np.max(temp_all_bleu_list)

        return best_score



class CrossSelectEvaluator(SentenceEvaluator):

    def __init__(self, split = "test", context_window=3, candidate_size = 20, delex = False):
        self.split = split
        self.candidate_size = candidate_size

        if delex:
            self.eval_dataset =  prepare_woz22_data_delex(split=self.split, context_window=context_window, num_of_candicate=candidate_size)
        else:
            self.eval_dataset =  prepare_woz22_data(split=self.split, context_window=context_window, only_system=True, num_of_candicate=candidate_size)

        self.queries, self.candidates, self.correct_answers = [] ,[],  []

        for item in self.eval_dataset:
            self.queries.append(" <turn_sep> ".join(item["context"]))
            self.candidates.append(item["candidates"])
            self.correct_answers.append(item["gold"])

        logger.info("Size of the dataset:\t{}".format(len(self.eval_dataset)))


        logger.info("Random baseline on "+  self.split + ": " + str(1/self.candidate_size))

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:

        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info(self.split + " evaluation" + out_txt)

        correct = 0
        total = 0
        problematic = 0

        queries = self.queries
        candidates = self.candidates
        correct_answers = self.correct_answers


        assert len(candidates) == len(queries)
        assert len(correct_answers) == len(queries)


        all_queries_candidate_pair_list = []
        for i in range(len(queries)):

            this_query = queries[i]
            this_candidate = candidates[i]
            this_answer = correct_answers[i]



            queries_candidate_pair_list = list(zip([this_query for _ in this_candidate], this_candidate))
            all_queries_candidate_pair_list.extend(queries_candidate_pair_list)
        all_scores = model.predict(all_queries_candidate_pair_list)
        score_list = []
        for i in range(0, len(candidates)):
            score_list.append(all_scores[self.candidate_size * i: self.candidate_size * (i + 1)])

        for i in range(len(queries)):
            this_query = queries[i]
            this_candidate = candidates[i]
            this_answer = correct_answers[i]
            # best_idx = get_best_idx(enc_queries[i], enc_candidates[i])
            gold_idx = get_gold_idx(this_candidate, this_answer)

            if (gold_idx == -1):
                problematic += 1
            total += 1
            best_idx = np.argmax(score_list[i])
            if gold_idx == best_idx:
                correct += 1

        r10 = 100 * (float(correct) / float(total))

        logger.info("R1-10 = {:s}".format(str(round(r10, 2))))

        return r10




if __name__ == '__main__':
    print(":D")