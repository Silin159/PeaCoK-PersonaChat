import sys
import logging
import os
import argparse
from transformers import BartTokenizer
import torch
import numpy as np
from torch.utils.data import TensorDataset
from pprint import pformat

from build_data import (
    create_data, build_dataloader, create_data_from_peacok_format
)


def init_config():
    parser = argparse.ArgumentParser(description='save_datasets')
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument('--seed', type=int, default=783435, metavar='S', help='random seed')
    parser.add_argument("--load_from", type=str, default=None,
                        help="save model")
    parser.add_argument('--revised', action='store_true', default=False, help='use revised')
    parser.add_argument('--cand', type=int, default=5, help='number of candidate')
    parser.add_argument("--max_history", type=int, default=4, help="length of dialogue context")


    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args = argparse.Namespace(**vars(args))
    return args


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    add_special_tokens = {
        'additional_special_tokens': [
            '<query>', '<response>', '<latent>', '<persona>', '<partner>'
        ]
    }
    args = init_config()
    if args.revised:
        data_from = "_revised"
    else:
        data_from = "_original"

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    format_str = logging.Formatter('%(asctime)s: %(levelname)s: %(message)s')
    logger.setLevel(level=logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(format_str)
    logger.addHandler(sh)
    logger.info(r"running %s" % ''.join(sys.argv))
    logger.info("Arguments: %s", pformat(args))

    logger.info("Get pretrained tokenizer")

    if args.load_from != None:
        tokenizer = BartTokenizer.from_pretrained(args.load_from)
    else:
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        num_added_toks = tokenizer.add_special_tokens(add_special_tokens)
        logger.info('We have added {} tokens'.format(num_added_toks))

    logger.info('We have {} tokens'.format(len(tokenizer)))

    logger.info("Build valid data")
    persona, persona_ext, partner, query, response, cand = create_data_from_peacok_format(
        "data/persona_peacok", dataset=args.dataset, mode="valid"
    )
    val_data = build_dataloader(
        persona, query, response, cand, tokenizer,
        partner_persona=partner,
        persona_ext=persona_ext,
        max_history=args.max_history, use_all=True
    )

    logger.info("Build train data")
    persona, persona_ext, partner, query, response, cand = create_data_from_peacok_format(
        "data/persona_peacok", dataset=args.dataset, mode="train"
    )
    train_data = build_dataloader(
        persona, query, response, cand, tokenizer,
        partner_persona=partner,
        persona_ext=persona_ext,
        max_history=args.max_history, n_cand=args.cand
    )

    MODEL_INPUTS = [
        "input_ids", "attention_mask", "lmlabels", "decoder_input_ids",
        "decoder_attention_mask", "cls_index", "clslabel"
    ]
    trainset = []
    valset = []
    for input_name in MODEL_INPUTS:
        if input_name == "clslabel":
            tensor = train_data[input_name].view(-1)
            logger.info("{}: {}".format(input_name, tensor.size()))
            trainset.append(tensor)
            tensor = val_data[input_name].view(-1)
            logger.info("{}: {}".format(input_name, tensor.size()))
            valset.append(tensor)
        else:
            tensor = train_data[input_name].view(-1, args.cand, train_data[input_name].size(-1))
            trainset.append(tensor)
            logger.info("{}: {}".format(input_name, tensor.size()))
            tensor = val_data[input_name].view(-1, 20, val_data[input_name].size(-1))
            valset.append(tensor)
            logger.info("{}: {}".format(input_name, tensor.size()))

    logger.info("Prepare dataloaders.")

    train_dataset = TensorDataset(*trainset)
    val_dataset = TensorDataset(*valset)

    os.makedirs(f'saved_datasets/{args.dataset}', exist_ok=True)
    import pickle
    with open(f'saved_datasets/{args.dataset}/train_dataset.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)
    with open(f'saved_datasets/{args.dataset}/val_dataset.pkl', 'wb') as f:
        pickle.dump(val_dataset, f)

    print('Done')