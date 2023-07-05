import sys
import logging
import os
import argparse
import numpy as np
import math
from pprint import pformat
import glob
import heapq
import shutil

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.metrics import Accuracy, Loss, MetricsLambda

from transformers import BartTokenizer, AdamW, WEIGHTS_NAME, CONFIG_NAME
from model.modeling_bart import BartForConditionalGeneration

# ------------------ Global accelerator object ------------------
from accelerate.logging import get_logger
from accelerate import Accelerator
accelerator = Accelerator(log_with='all')
# ---------------------------------------------------------------

def delete_nth_latest_checkpoint(output_dir, n):
    checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint*'))
    if len(checkpoints) < n:
        print(f"Not enough 'checkpoint' files in {output_dir} to delete the {n}-th latest.")
        return
    latest_checkpoints = heapq.nlargest(n, checkpoints, key=os.path.getctime)
    checkpoint_to_delete = latest_checkpoints[-1]
    try:
        shutil.rmtree(checkpoint_to_delete)
        print(f"Successfully deleted the {n}-th latest 'checkpoint' file: {checkpoint_to_delete}")
    except OSError as e:
        print(f"Error deleting the file: {checkpoint_to_delete}. Error: {e}")

def init_config():
    parser = argparse.ArgumentParser(description='BART')
    parser.add_argument('--peft', action='store_true', default=False)
    parser.add_argument('--include-val', action='store_true', default=False)
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--n_saved', type=int, default=5)
    parser.add_argument('--world-size', type=int, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=783435, metavar='S', help='random seed')
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for dialogue training")
    parser.add_argument("--val_batch_size", type=int, default=4, help="Batch size for dialogue training")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--warmup", type=float, default=0.2, help="warmup rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--load_from", type=str, default=None, help="save model")
    parser.add_argument('--revised', action='store_true', default=False, help='use revised')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Accumulate gradients on several steps")
    parser.add_argument('--cand', type=int, default=5, help='number of candidate')
    parser.add_argument("--max_history", type=int, default=4, help="length of dialogue context")

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    args = argparse.Namespace(**vars(args))
    return args

def truncate_tensor_dataset(dataset, new_len=256):
    truncated = []
    for i, s in enumerate(dataset):
        truncated.append(s)
        if i == new_len: break
    truncated = TensorDataset(*[torch.stack(t) for t in zip(*truncated)])
    return truncated

def get_parameter_devices(model):
    devices = set()
    for param in model.parameters():
        devices.add(param.device)
    return devices


# ------------------ Single training pass ------------------
def train_step(engine, batch):
    model = engine.state.model
    optimizer = engine.state.optimizer
    args = engine.state.args
    accelerator = engine.state.accelerator

    model.train()

    (
        input_ids, attention_mask, lmlabels,
        decoder_input_ids, decoder_attention_mask,
        cls_index, clslabel,
        *_
    ) = batch

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=lmlabels,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        cls_index=cls_index,
        clslabel=clslabel
    )    

    (mlm_loss, cls_loss) = outputs.loss
    (mlm_loss, cls_loss) = (mlm_loss.mean(), cls_loss.mean())
    loss = mlm_loss + cls_loss
    accelerator.backward(loss)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

    if engine.state.iteration % args.gradient_accumulation_steps == 0:
        optimizer.step()
        model.zero_grad()

    return {
        'loss': loss.item(), 'cls_loss': cls_loss.item(), 'mlm_loss': mlm_loss.item()
    }


# ------------------ Single eval pass ------------------
def eval_step(engine, batch):
    model = engine.state.model
    accelerator = engine.state.accelerator

    model.eval()

    with torch.no_grad():
        batch = tuple(input_tensor.to(accelerator.device) for input_tensor in batch)
        (
            input_ids, attention_mask, lmlabels,
            decoder_input_ids, decoder_attention_mask,
            cls_index, clslabel,
            *_
        ) = batch

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=lmlabels,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            cls_index=cls_index,
            clslabel=clslabel
        )
        lm_logits, cls_logits = outputs.logits

        lm_logits = lm_logits.view(-1, lm_logits.size(-1))
        lmlabels = lmlabels.view(-1)

        return {
            'lm_logits': lm_logits, 'lm_labels': lmlabels,
            'cls_logits': cls_logits, 'cls_label': clslabel
        }


# ================== Main training process ==================
def main():
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

    args.output_dir = f"{args.dataset}{data_from}"
    if not os.path.exists(args.output_dir) and accelerator.is_main_process:
        os.makedirs(args.output_dir)
    log_file = os.path.join(args.output_dir, "train.log")


    # ------------------ Logger ------------------

    logger = get_logger(__name__, log_level='INFO')

    if accelerator.is_main_process:
        logging.basicConfig(
            format='%(asctime)s: %(levelname)s: %(message)s',
            datefmt="[%H:%M:%S]",
        )
    format_str = logging.Formatter('%(asctime)s: %(levelname)s: %(message)s')
    fh = logging.FileHandler(filename=log_file, encoding='utf-8', mode='w')
    fh.setFormatter(format_str)
    logger.logger.addHandler(fh)

    logger.info(r"running %s" % ''.join(sys.argv), main_process_only=True)
    logger.info("Arguments: %s", pformat(args), main_process_only=True)


    # ------------------ Model and Tokenizer ------------------

    logger.info("Get pretrained model and tokenizer", main_process_only=True)
    if args.load_from != None:
        tokenizer = BartTokenizer.from_pretrained(args.load_from)
        model = BartForConditionalGeneration.from_pretrained(args.load_from)
    else:
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        num_added_toks = tokenizer.add_special_tokens(add_special_tokens)
        logger.info('We have added {} tokens'.format(num_added_toks), main_process_only=True)

        model = BartForConditionalGeneration.from_pretrained(
            "facebook/bart-large", num_labels=1
        )
        model.resize_token_embeddings(len(tokenizer))
        model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids('<response>')
        model.config.forced_bos_token_id = None

    logger.info('We have {} tokens'.format(len(tokenizer)), main_process_only=True)
    logger.info("Complete loading model.", main_process_only=True)

    # ------------------ PEFT ------------------

    if args.peft:
        from peft import get_peft_model, TaskType, LoraConfig

        logger.info("Performing PEFT", main_process_only=True)
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
        )
        
        model = get_peft_model(model, peft_config)
        if accelerator.is_main_process:
            model.print_trainable_parameters()

    # ------------------ Datasets and Dataloaders ------------------

    logger.info("Loading datasets.", main_process_only=True)
    import pickle

    with open(f'saved_datasets/{args.dataset}/train_dataset.pkl', 'rb') as f:
        train_dataset = pickle.load(f)
    with open(f'saved_datasets/{args.dataset}/val_dataset.pkl', 'rb') as f:
        eval_dataset = pickle.load(f)

    logger.info("Preparing dataloaders.", main_process_only=True)
    train_loader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=0
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.val_batch_size, shuffle=False
    )
    logger.info("Dataloaders ready.", main_process_only=True)


    # ------------------ Optimizer and Scheduler ------------------

    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)
    scheduler = PiecewiseLinear(
        optimizer, "lr",
        [(0, args.lr), (args.epochs * len(train_loader), 0.0)]
    )

    # ------------------ HF Accelerate ------------------
    
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    # ------------------ Ignite Engine Setup ------------------

    trainer = Engine(train_step)
    evaluator = Engine(eval_step)

    trainer.state.model = evaluator.state.model = model
    trainer.state.args = evaluator.state.args = args
    trainer.state.optimizer = optimizer
    trainer.state.accelerator = evaluator.state.accelerator = accelerator

    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    RunningAverage(output_transform=lambda x: x['loss']).attach(trainer, 'loss')
    RunningAverage(output_transform=lambda x: x['cls_loss']).attach(trainer, 'cls_loss')
    RunningAverage(output_transform=lambda x: x['mlm_loss']).attach(trainer, 'mlm_loss')

    metrics = {}
    metrics["nll"] = Loss(
        torch.nn.CrossEntropyLoss(ignore_index=-100),
        output_transform=lambda x: (x['lm_logits'], x['lm_labels'])
    )
    metrics["ppl"] = MetricsLambda(math.exp, metrics['nll'])
    metrics["accuracy"] = Accuracy(
        output_transform=lambda x: (x['cls_logits'], x['cls_label'])
    )
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    def write_metrics(engine, accelerator):
        metrics = engine.state.metrics
        if accelerator.is_main_process:
            logger.info("Validation: %s" % pformat(metrics))
        
    evaluator.add_event_handler(
        Events.COMPLETED,
        lambda __: write_metrics(evaluator, accelerator)
    )

    # Progress bar updates are done on the main process only
    if accelerator.is_main_process:
        pbar = ProgressBar(position=0, persist=True, ncols=140)
        pbar.attach(trainer, metric_names=['loss', 'cls_loss', 'mlm_loss'])
        pbar_eval = ProgressBar(persist=True, ncols=140)
        pbar_eval.attach(evaluator)
    
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        lambda engine: logger.info(f"Complete trainer epoch: {engine.state.epoch}")
    )

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        lambda _: evaluator.run(eval_loader)
    )

    
    # ------------------ Checkpointing ------------------

    # Checkpointing is done on the main process only
    log_dir = args.output_dir

    def checkpoint_handler_wrapper(accelerator, engine, model, output_dir):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            delete_nth_latest_checkpoint(output_dir, n=args.n_saved)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            f"{output_dir}/checkpoint_{engine.state.epoch}_lr_{optimizer.param_groups[0]['lr']}"
        )
        unwrapped_model.save_pretrained(f'{output_dir}/checkpoint_latest')

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        lambda _: checkpoint_handler_wrapper(
            accelerator, engine=trainer, model=model, output_dir=args.output_dir
        )
    )

    torch.save(args, log_dir + '/model_training_args.bin')
    getattr(model, 'module', model).config.to_json_file(os.path.join(log_dir, CONFIG_NAME))
    tokenizer.save_pretrained(log_dir)


    # ------------------ !!! Run the training !!! ------------------
    logger.info("Begin training")
    trainer.run(train_loader, max_epochs=args.epochs)


if __name__ == '__main__':
    main()
