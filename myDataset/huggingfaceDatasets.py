import json
from datasets import load_dataset
from itertools import chain
import random


def prepare_woz22_data(split, context_window=None, only_system=False, only_user=False, num_of_candicate=10):
    random.seed(42)

    dataset = load_dataset("multi_woz_v22")

    if split == "dev":
        split = "validation"
    training_dials = dataset[split]["turns"]
    data_samples = []

    allUtts = list(chain(*(list(map(lambda x: x["utterance"], training_dials)))))

    cur_idx = 0

    for dial in training_dials:
        utts = dial["utterance"]
        speaker = dial["speaker"]
        for i in range(len(utts)):

            sentc = utts[0 if context_window is None else i - context_window:i]
            sentr = utts[i]
            assert (allUtts[cur_idx]) == sentr

            sampled_candicates = random.sample(allUtts, k=num_of_candicate)

            if sentr in sampled_candicates:
                sampled_candicates.remove(sentr)

            temp_candicates = [sentr] + sampled_candicates
            temp_candicates = temp_candicates[:num_of_candicate]

            assert temp_candicates[0] == sentr
            cur_idx += 1
            temp_dic = {}
            temp_dic["context"] = sentc
            temp_dic["candidates"] = temp_candicates
            temp_dic["gold"] = sentr

            if (only_system and speaker[i] == 1):
                data_samples.append(temp_dic)
            if (only_user and speaker[i] == 0):
                data_samples.append(temp_dic)
            if (not only_user and not only_system):
                data_samples.append(temp_dic)

    return data_samples


if __name__ == '__main__':
    print(":D")