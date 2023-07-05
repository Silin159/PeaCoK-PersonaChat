import sys
import logging
import os
import argparse
import numpy as np
import math
from pprint import pformat

import torch
from torch.utils.data import DataLoader, TensorDataset

from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar
from ignite.metrics import Accuracy, Loss, MetricsLambda

from transformers import BartTokenizer
from model.modeling_bart import BartForConditionalGeneration

# ------------------ Single eval pass ------------------
from train import eval_step

# ------------------ Global accelerator object ------------------
from accelerate.logging import get_logger
from accelerate import Accelerator
accelerator = Accelerator(log_with='all')
# ---------------------------------------------------------------

def truncate_tensor_dataset(dataset, new_len=256):
    truncated = []
    for i, s in enumerate(dataset):
        truncated.append(s)
        if i == new_len: break
    truncated = TensorDataset(*[torch.stack(t) for t in zip(*truncated)])
    return truncated


def init_config():
    parser = argparse.ArgumentParser(description='BART')
    parser.add_argument('--peft', action='store_true', default=False)
    parser.add_argument("--dataset", type=str, default="", help="save model")
    parser.add_argument("--model_checkpoint", type=str, default=None, help="save model")
    parser.add_argument("--tokenizer_checkpoint", type=str, default="facebook/bart-large")
    parser.add_argument('--seed', type=int, default=783435, metavar='S', help='random seed')
    parser.add_argument("--eval_batch_size", type=int, default=2, help="Batch size for evaluation")
    parser.add_argument('--revised', action='store_true', default=False, help='use revised')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args = argparse.Namespace(**vars(args))
    return args


# ================== Main evaluation process ==================
def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    args = init_config()

    # ------------------ Logger ------------------

    # logger = logging.Logger(name=__file__, level=logging.INFO)
    # logging.basicConfig(
    #     format='%(asctime)s: %(levelname)s: %(message)s',
    #     datefmt="[%H:%M:%S]",
    # )

    class Log:
        def info(self, *args):
            print(*args)
    
    logger = Log()

    logger.info(r"running %s" % ''.join(sys.argv))
    logger.info("Arguments: %s", pformat(args))


    # ------------------ PEFT Model and Tokenizer ------------------

    logger.info("Get pretrained model and tokenizer")

    if args.peft:
        from peft import PeftModel, PeftConfig
        config = PeftConfig.from_pretrained(args.model_checkpoint)
        tokenizer = BartTokenizer.from_pretrained(config.base_model_name_or_path)
        model = BartForConditionalGeneration.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, args.model_checkpoint)
    else:
        tokenizer = BartTokenizer.from_pretrained(args.tokenizer_checkpoint)
        model = BartForConditionalGeneration.from_pretrained(args.model_checkpoint)

    model = model.to(accelerator.device)

    logger.info('We have {} tokens'.format(len(tokenizer)))
    logger.info("Complete loading model.")

    # ------------------ Datasets and Dataloaders ------------------

    logger.info("Loading dataset.")
    import pickle

    with open(f'saved_datasets/{args.dataset}/val_dataset.pkl', 'rb') as f:
        eval_dataset = pickle.load(f)
        # eval_dataset = truncate_tensor_dataset(eval_dataset, 256)

    logger.info("Preparing dataloader.")
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.eval_batch_size, shuffle=False
    )
    logger.info("Dataloader ready.")

    # ------------------ Ignite Engine Setup ------------------

    evaluator = Engine(eval_step)

    evaluator.state.model = model
    evaluator.state.args = args
    evaluator.state.accelerator = accelerator

    metrics = {}
    metrics["nll"] = Loss(
        torch.nn.CrossEntropyLoss(ignore_index=-100),
        output_transform=lambda x: (x['lm_logits'], x['lm_labels'])
    )
    metrics["ppl"] = MetricsLambda(math.exp, metrics['nll'])

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    pbar_eval = ProgressBar(persist=True, ncols=140)
    pbar_eval.attach(evaluator)

    def write_metrics(engine):
        metrics = engine.state.metrics
        logger.info("Validation: %s" % pformat(metrics))
        log_file = os.path.join(args.model_checkpoint, "ppl.txt")
        with open(log_file, 'w') as f:
            f.write(f"Validation: {metrics}\n")
        
    evaluator.add_event_handler(
        Events.COMPLETED,
        lambda __: write_metrics(evaluator)
    )

    # ------------------ !!! Run the evaluation !!! ------------------
    logger.info("Begin evaluation.")
    evaluator.run(eval_loader)


if __name__ == '__main__':
    main()


