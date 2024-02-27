import random

from sentence_transformers import LoggingHandler
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader

from sentence_transformers.cross_encoder import CrossEncoder

import math

from myDataset import *
from src.utils import CrossSelectEvaluator

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

def process_data(context_window=3):


    data = prepare_woz22_data_delex(split="train", context_window = context_window, num_of_candicate= 20)

    data_samples = []
    for entry in data:

        gold = entry["gold"]
        resp_candidates = entry["candidates"]
        context = entry["context"]

        sentc = str(" <turn_sep> ".join(context[0 if context_window is None else -context_window:]))
        sentr = gold

        assert (sentr) == resp_candidates[0]
        data_samples.append(InputExample(texts=[sentc, sentr], label=1))
        for item in resp_candidates[1:]:
            data_samples.append(InputExample(texts=[sentc, item], label=0))

    return data_samples



'''
Main function here
'''


def run_experiment():

    batch_size = 64
    epochs = 3
    context_window = 3
    init_model_path = "google-bert/bert-base-uncased"
    # init_model_path = "cross-encoder/quora-distilroberta-base"

    note = "delex"


    save_model_path = init_model_path.replace("/", "-")
    output_path = "./output/selection_" + save_model_path + "_" + note+ "/"

    train_samples = process_data(context_window=context_window)
    assert len(train_samples) > 0

    model = CrossEncoder(init_model_path, num_labels=1)

    model.max_seq_length = 128

    special_tokens = Vocab().special_tokens
    model.tokenizer.add_tokens(special_tokens, special_tokens=True)
    model.model.resize_token_embeddings(len(model.tokenizer))

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)

    evaluator = CrossSelectEvaluator(split="dev", context_window = context_window, delex=True)
    warmup_steps = math.ceil(len(train_dataloader) * epochs / batch_size * 0.1)

    model.fit(train_dataloader=train_dataloader,
              evaluator=evaluator,
              epochs=epochs,
              warmup_steps=warmup_steps,
              output_path=output_path)


# Main program starts here
def main():
    run_experiment()

## MAIN starts here
if __name__ == '__main__':
    main()
