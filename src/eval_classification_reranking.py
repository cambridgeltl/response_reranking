import logging
from sentence_transformers import CrossEncoder
from src.utils import CrossOverGenBleuEvaluator

from sentence_transformers import LoggingHandler
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

def run_eval():

    init_model_path ="./output/reranking_classification_0.7_mpnet/"

    model = CrossEncoder(init_model_path)
    evaluator = CrossOverGenBleuEvaluator(data_path= "./data/0.7_test.json", context_window = 3)
    evaluator.__call__(model)

# Main program starts here
def main():
    run_eval()

## MAIN starts here
if __name__=='__main__':
    main()
