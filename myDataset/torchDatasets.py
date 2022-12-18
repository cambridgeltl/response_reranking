import json
import logging
import glob
import itertools
import random

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
}
SPECIAL_TOKENS_VALUES = ["<bos>", "<eos>", "<pad>"]


def load_damd_lex_delex_map(path="./data/multi-woz-processed/data_for_damd.json"):
    f = open(path)
    dataset = json.load(f)
    f.close()

    utt_lex_delex_map = {}

    for key in list(dataset.keys())[:]:
        dial = dataset[key]
        for utt in dial["log"]:
            user_delex = utt["user_delex"]
            user_lex = utt["user"]
            sys_delex = utt["resp"]
            sys_lex = utt["resp_nodelex"]
            utt_lex_delex_map[user_lex] = user_delex
            utt_lex_delex_map[sys_lex] = sys_delex

    return utt_lex_delex_map

class MinLTOutputDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path

        temp_file = open(file_path, 'r')
        count = 0

        temp_responses = []
        for line in temp_file:
            count += 1
            temp_responses.append(line.strip())
        temp_file.close()
        temp_responses = temp_responses[1:]
        temp_responses = list(filter(lambda x: ":" in x, temp_responses))

        gold_output_pair_list = list(zip(temp_responses, temp_responses[1:]))[::2]

        gold_output_pair_list = list(map(lambda x: (x[0][6:], x[1][6:]), gold_output_pair_list))
        gold_output_pair_list = list(filter(lambda x: len(x[0]) > 0 or len(x[1]) > 0, gold_output_pair_list))
        self.gold_output_pair_list = gold_output_pair_list
        self.gold_output_pair_list = list(
            map(lambda x: (x[0], [x[1]], gold_output_pair_list), self.gold_output_pair_list))

    def __getitem__(self, index):
        return self.gold_output_pair_list[index]

    def __len__(self):
        return len(self.gold_output_pair_list)


class MinLTOverGenDataset(Dataset):
    # [(context, [candidates])]
    def __init__(self, file_path, isLexicalised = False, context_window = 3):

        self.file_path = file_path
        self.gold_output_pair_list = []
        self.context_window = context_window
        self.isLexicalised = isLexicalised
        self.lex_delex_map = load_damd_lex_delex_map()

        with open(self.file_path) as jsonFile:
            jsonObject = json.load(jsonFile)
            jsonFile.close()

        for item in jsonObject:


            if self.isLexicalised:
                ground_truth_resp = item["resp_nodelex"].replace(" <eos_r>", "")
            else:
                ground_truth_resp = item["resp_text"]

            context = " <turn_sep> ".join(list(map(lambda x : self.lex_delex_map[x], item["context_text"][-self.context_window:])))


            if ground_truth_resp != "":
                sentn_list = list(map(lambda x: x.replace(" <eos_r>", ""), item[ "over_gen"]))
                sentn_list = list(map(lambda x: x.replace(".", " ."), sentn_list))
                sentn_list = list(map(lambda x: x.replace("?", " ?"), sentn_list))
                self.gold_output_pair_list.append(
                    (ground_truth_resp, context, sentn_list ))

    def __getitem__(self, index):
        return self.gold_output_pair_list[index]

    def __len__(self):
        return len(self.gold_output_pair_list)

class DummyMinLTOverGenDataset(Dataset):
    # [(context, [candidates])]
    def __init__(self, file_path, isLexicalised = False, context_window = 3):

        self.file_path = file_path
        self.gold_output_pair_list = []
        self.context_window = context_window
        self.isLexicalised = isLexicalised
        self.lex_delex_map = load_damd_lex_delex_map()

        with open(self.file_path) as jsonFile:
            jsonObject = json.load(jsonFile)
            jsonFile.close()

        for item in jsonObject:


            if self.isLexicalised:
                ground_truth_resp = item["resp_nodelex"].replace(" <eos_r>", "")
            else:
                ground_truth_resp = item["resp_text"]

            context = " <turn_sep> ".join(list(map(lambda x : self.lex_delex_map[x], item["context_text"][-self.context_window:])))

            greedy_gen = [item["resp_gen"]]

            greedy_gen = list(map(lambda x: x.replace(" <eos_r>", ""), greedy_gen))
            greedy_gen = list(map(lambda x: x.replace(".", " ."), greedy_gen))
            greedy_gen = list(map(lambda x: x.replace("?", " ?"), greedy_gen))
            greedy_gen = greedy_gen[0]
            if ground_truth_resp != "":

                sentn_list = list(map(lambda x: x.replace(" <eos_r>", ""), item[ "over_gen"]))
                sentn_list = list(map(lambda x: x.replace(".", " ."), sentn_list))
                sentn_list = list(map(lambda x: x.replace("?", " ?"), sentn_list))
                self.gold_output_pair_list.append(
                    (ground_truth_resp, context, sentn_list, greedy_gen))

    def __getitem__(self, index):
        return self.gold_output_pair_list[index]

    def __len__(self):
        return len(self.gold_output_pair_list)


def clean_text(utt: str):
    ans = utt.replace("\n", "").strip().lower()
    ans = ans.replace("\t", " ").strip().lower()
    while "  " in ans:
        ans = ans.replace("  ", " ")
    return ans

def get_context_response_pair_list(dialogue):
    ans = []

    for i in range(len(dialogue["turns"])):
        temp = {}
        temp["context"] = list(
            map(lambda x: (clean_text(x["utterance"]), x["speaker"]), dialogue["turns"][:i]))
        temp["utt"] = clean_text(dialogue["turns"][i]["utterance"])
        temp["speaker"] = dialogue["turns"][i]["speaker"]
        #         temp["act"] = dialogue["turns"][i]["dialog_act"]
        ans.append(temp)
    return ans

def prepare_woz22_data_delex(split, context_window=None, num_of_candicate=10):
    random.seed(42)

    if split == "dev":
        split = "validation"

    if split == "train":
        path = "./data/0.7_train.json"
    elif split == "validation":
        path = "./data/0.7_dev.json"
    else:
        path = "./data/0.7_test.json"

    f = open(path)
    dataset = json.load(f)
    f.close()

    lex_delex_utt_map = load_damd_lex_delex_map()

    data_samples = []
    allUtts = list((list(map(lambda x: x["resp_text"], dataset))))

    for entry in dataset:
        context_lex = entry["context_text"]
        context_delex = list(map(lambda x: lex_delex_utt_map[x], context_lex))
        sentc = context_delex
        sentr = entry["resp_text"]
        sampled_candicates = random.sample(allUtts, k=num_of_candicate)

        if sentr in sampled_candicates:
            sampled_candicates.remove(sentr)

        temp_candicates = [sentr] + sampled_candicates
        temp_candicates = temp_candicates[:num_of_candicate]

        assert temp_candicates[0] == sentr
        temp_dic = {}
        temp_dic["context"] = sentc
        temp_dic["candidates"] = temp_candicates
        temp_dic["gold"] = sentr

        data_samples.append(temp_dic)

    return data_samples

if __name__ == '__main__':
    print(":D")