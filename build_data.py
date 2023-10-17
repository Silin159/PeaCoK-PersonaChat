import random
from tqdm import tqdm
import torch
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import json
import os


def get_token_id(tokenizer):
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    sep_id = tokenizer.sep_token_id
    query_id, res_id, latent_id, persona_id, partner_id = tokenizer.convert_tokens_to_ids(
        ['<query>', '<response>', '<latent>', '<persona>', '<partner>']
    )
    return bos_id, eos_id, pad_id, sep_id, query_id, res_id, latent_id, persona_id, partner_id

def create_data(data_file):
    with open(data_file, "r", encoding="utf8") as f:
        persona =[]
        query = []
        response = []
        cand = []
        is_persona = False
        tmp_persona = []
        tmp_query = []
        tmp_response = []
        tmp_cand = []
        first = True
        cnt = 0
        sum_u = 0
        for line in f:
            cnt += 1
            line = line.strip()
            if "your persona: " in line:
                if not is_persona and not first:
                    query.append(tmp_query)
                    response.append(tmp_response)
                    cand.append(tmp_cand)
                    sum_u += len(tmp_query)
                    tmp_query = []
                    tmp_response = []
                    tmp_cand = []
                first = False
                is_persona = True
                line = line.split(": ", maxsplit=1)[1]
                tmp_persona.append(line)
            else:
                if is_persona:
                    persona.append(tmp_persona)
                    is_persona = False
                    tmp_persona = []
                line = line[line.find(" ")+1:]
                tmp_query.append(line.split("\t")[0])
                tmp_response.append(line.split("\t")[1])
                tmp_cand.append(line.split("\t")[3].split("|"))
        query.append(tmp_query)
        response.append(tmp_response)
        cand.append(tmp_cand)
        sum_u += len(tmp_query)
        assert len(query) == len(response) == len(persona) == len(cand)

    print("{} has {} dialog and {} query".format(data_file, len(query), sum_u))

    return persona, query, response, cand

def create_data_from_peacok_format(
    data_dir,
    filename=None,
    dataset='persona_chat',
    mode='train'
    ):
    """
    NOTE - This function does NOT expect the same format for the input file 
    as does the create_data function! This function expects the input file 
    to be in the format of the Persona-PeaCoK dataset:

    dialogue_id: {
        "persona1_ori": [persona1_ori_1, persona1_ori_2, ...],  
        "persona2_ori": [persona2_ori_1, persona2_ori_2, ...],  
        "persona1_ext": [persona1_ext_1, persona1_ext_2, ...],
        "persona2_ext": [persona2_ext_1, persona2_ext_2, ...],
        "text": [query_1, response_1, query_2, response_2, ...],
        "text_plain_cands": [TBD]
    }

    where the persona1_ori and persona2_ori are the original personas, 
    and the persona1_ext and persona2_ext are the extended personas 
    - i.e. extended using the PeaCoK dataset.

    Returns: the expected format for the build_dataloaders function.
    (4 lists: persona, query, response, cand)
    """

    if filename is not None:
        filepath = os.path.join(data_dir, filename)
    else:
        inferred_filename = {
            'persona_chat': '{}_persona_original_chat_convai2.json',
            'persona_chat_peacok': '{}_persona_original_chat_ext.json',
            'persona_chat_peacok_retrieved': '{}_persona_original_chat_ext_retrieved.json',
            'persona_chat_peacok_induced': '{}_persona_original_chat_ext_induced.json',
            'persona_chat_peacok_random_induced': '{}_persona_original_chat_ext_random_induced.json',
            'persona_chat_peacok_retrieved_induced': '{}_persona_original_chat_ext_retrieved_induced.json',
            }
        filepath = os.path.join(data_dir, inferred_filename[dataset].format(mode))
        print('FILEPATH: ', filepath)

    data_dict = json.load(open(filepath))
    persona, persona_ext, partner_persona, query, response, cand = [], [], [], [], [], []
    for dialogue_id, sample in data_dict.items():
        # Extending the current persona
        if dataset == 'persona_chat':
            persona.append(sample['persona1_ori'])
        elif dataset in ['persona_chat_peacok', 'persona_chat_peacok_retrieved']:
            persona.append(sample['persona1_ori'] + sample['persona1_ext'])
        elif dataset in ['persona_chat_peacok_induced']:
            persona.append(sample['persona1_ori'])
            partner_persona.append(sample['persona2_ext'])
        elif dataset in ['persona_chat_peacok_random_induced']:
            persona.append(sample['persona1_ori'] + sample['persona1_ext'])
            partner_persona.append(sample['persona2_ext'])
        elif dataset in ['persona_chat_peacok_retrieved_induced']:
            persona.append(sample['persona1_ori'])
            persona_ext.append(sample['persona1_ext'])
            partner_persona.append(sample['persona2_ext'])
        # Queries - even utterances
        query.append(sample['text'][0::2])
        # Responses - odd utterances
        response.append(sample['text'][1::2])
        # Candidates
        cand.append([
            current_text_plain_cands.split("\t")[3].split("|")
            for current_text_plain_cands in sample['text_plain_cands']
        ])
    return persona, persona_ext, partner_persona, query, response, cand


def create_encoder_input(
    per,
    partner,
    history,
    query_id, res_id, latent_id, persona_id, partner_id, sep_id, eos_id
    ):
    encoder_input_ids = []

    per_input_ids = [latent_id] + [persona_id]
    for x in per:
        per_input_ids += x + [sep_id]
    
    partner_input_ids = [partner_id]
    for x in partner:
        partner_input_ids += x + [sep_id]

    encoder_input_ids += per_input_ids + partner_input_ids

    for i in range(len(history)):
        if i % 2 == 0:
            encoder_input_ids += [query_id] + history[i] + [eos_id]
        else:
            encoder_input_ids += [res_id] + history[i] + [eos_id]
    attention_mask = [1] * len(encoder_input_ids)
    per_attention_mask = [1] * len(per_input_ids)

    return encoder_input_ids, attention_mask, per_input_ids, per_attention_mask

def create_decoder_input(response_ids, res_id, eos_id, golden=None):
    assert golden != None

    decoder_lmlabel = response_ids + [eos_id]
    decoder_input_ids = [res_id] + response_ids
    decoder_cls_index = [-100] * (len(decoder_lmlabel) - 1) + [eos_id]
    decoder_attention_mask = [1] * len(decoder_input_ids)


    if golden == False:
        decoder_lmlabel = [-100] * len(decoder_lmlabel)

    assert len(decoder_lmlabel) == len(decoder_input_ids)

    return decoder_lmlabel, decoder_input_ids, decoder_cls_index, decoder_attention_mask



def build_dataloader(
    persona, query, response, cand, tokenizer,
    partner_persona=None, persona_ext=None,
    max_history=4, n_cand=5, use_all=False
    ):
    partner_persona = None if len(partner_persona) == 0 else partner_persona
    persona_ext = None if len(persona_ext) == 0 else persona_ext

    bos_id, eos_id, pad_id, sep_id, query_id, res_id, latent_id, persona_id, partner_id = \
        get_token_id(tokenizer)

    num_dialogues = len(query)
    dataset = defaultdict(list)
    for i in range(num_dialogues):
        persona_ = [] if len(persona) == 0 else persona[i]
        per_list = []
        for per in persona_:
            persona_ids = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(per, add_prefix_space=True)
            )
            per_list.append(persona_ids)
        partner_persona_ = None if partner_persona is None else partner_persona[i]
        persona_ext_ = None if persona_ext is None else persona_ext[i]
        query_ = query[i]
        response_ = response[i]
        cand_ = cand[i]
        history = []
        assert len(query_) == len(response_)

        # Iterating over turns of dialogue history
        for j in range(len(query_)):
            if use_all:
                noise_candidate = cand_[j][:-1]
            else:
                noise_candidate = random.sample(cand_[j][:-1], n_cand-1)

            # Take the induced persona from turn j and format the same way as per_list
            tmp_partner_per_list = []
            if partner_persona is not None:
                for partner_per in partner_persona_[str(j)]:
                    partner_persona_ids = tokenizer.convert_tokens_to_ids(
                        tokenizer.tokenize(partner_per, add_prefix_space=True)
                    )
                    tmp_partner_per_list.append(partner_persona_ids)
            
            # Take the induced persona from turn j and format the same way as per_list
            tmp_per_ext_list = []
            if persona_ext is not None:
                for per_ext in persona_ext_[str(j)]:
                    persona_ext_ids = tokenizer.convert_tokens_to_ids(
                        tokenizer.tokenize(per_ext, add_prefix_space=True)
                    )
                    tmp_per_ext_list.append(persona_ext_ids)

            query_ids = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(query_[j], add_prefix_space=True)
            )
            response_ids = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(response_[j], add_prefix_space=True)
            )

            noise_cand_ids_list = [
                tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text, add_prefix_space=True))
                for text in noise_candidate
            ]

            history.append(query_ids)
            history.append(response_ids)
            tmp_history = history[-2 * max_history: -1]

            encoder_input_ids, attention_mask, per_input_ids, per_attention_mask = \
                create_encoder_input(
                    per_list + tmp_per_ext_list, # LIST OF TOKEN IDS OF SINGLE PERSONA FACTS
                    tmp_partner_per_list,
                    tmp_history, # LAST M (Q, R) PAIRS WITHOUT R_last (AT TURN J !!!)
                    query_id, res_id, latent_id, persona_id, partner_id, sep_id, eos_id # Special tokens
                    )

            decoder_lmlabel, decoder_input_ids, decoder_cls_idx, decoder_attention_mask = \
                create_decoder_input(
                    response_ids, # LIST OF TOKEN IDS OF CURRENT RESPONSE (AT TURN J !!!)
                    res_id, eos_id, golden=True
                    )

            dataset["input_ids"].append(encoder_input_ids)
            dataset["attention_mask"].append(attention_mask)
            dataset["lmlabels"].append(decoder_lmlabel)
            dataset["decoder_input_ids"].append(decoder_input_ids)
            dataset["decoder_attention_mask"].append(decoder_attention_mask)
            dataset["cls_index"].append(decoder_cls_idx)
            dataset["clslabel"].append([0])
            for k in range(len(noise_cand_ids_list)):
                decoder_lmlabel, decoder_input_ids, decoder_cls_idx,\
                    decoder_attention_mask = create_decoder_input(noise_cand_ids_list[k], res_id, eos_id, golden=False)
                dataset["input_ids"].append(encoder_input_ids)
                dataset["attention_mask"].append(attention_mask)
                dataset["lmlabels"].append(decoder_lmlabel)
                dataset["decoder_input_ids"].append(decoder_input_ids)
                dataset["decoder_attention_mask"].append(decoder_attention_mask)
                dataset["cls_index"].append(decoder_cls_idx)


    for item_name, item in dataset.items():
        if item_name == "input_ids":
            item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                              batch_first=True, padding_value=pad_id)

            dataset[item_name] = item
        elif item_name == "lmlabels":
            item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                batch_first=True, padding_value=-100)
            dataset[item_name] = item
        elif item_name == "attention_mask" or item_name == "decoder_attention_mask" or item_name == "per_attention_mask":
            item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                batch_first=True, padding_value=0)
            dataset[item_name] = item
        elif item_name == "decoder_input_ids":
            item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                batch_first=True, padding_value=pad_id)
            dataset[item_name] = item
        elif item_name == "clslabel":
            dataset[item_name] = torch.tensor(item).view(-1,1)
        elif item_name == "cls_index":
            item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                batch_first=True, padding_value=-100)
            dataset[item_name] = item

    return dataset

