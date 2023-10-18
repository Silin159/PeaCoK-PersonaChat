import os
import json
import copy

def extend_with_retrieved(persona_chat, induced_and_retrieved):
    persona_chat = copy.deepcopy(persona_chat)
    persona_chat_extended_retrieved = {}
    for dialogue_id in persona_chat.keys():
        persona_chat_extended_retrieved[dialogue_id] = {}
        persona_chat[dialogue_id]['persona1_ext'] = \
            induced_and_retrieved[dialogue_id]['relevant_speaker_personas']
        keys_to_keep = [
            'persona1_ori', 'persona1_ext', 'text', 'text_plain', 'text_plain_cands'
        ]
        for k in keys_to_keep:
            persona_chat_extended_retrieved[dialogue_id][k] = persona_chat[dialogue_id][k]
    return persona_chat_extended_retrieved

def extend_with_induced(persona_chat, induced_and_retrieved):
    persona_chat = copy.deepcopy(persona_chat)
    persona_chat_extended_retrieved = {}
    for dialogue_id in persona_chat.keys():
        persona_chat_extended_retrieved[dialogue_id] = {}
        persona_chat[dialogue_id]['persona2_ext'] = \
            induced_and_retrieved[dialogue_id]['induced_partner_personas']
        keys_to_keep = [
            'persona1_ori', 'persona2_ext', 'text', 'text_plain', 'text_plain_cands'
        ]
        for k in keys_to_keep:
            persona_chat_extended_retrieved[dialogue_id][k] = persona_chat[dialogue_id][k]
    return persona_chat_extended_retrieved

def extend_with_random_induced(persona_chat, induced_and_retrieved):
    persona_chat = copy.deepcopy(persona_chat)
    persona_chat_extended_random_induced = {}
    for dialogue_id in persona_chat.keys():
        persona_chat_extended_random_induced[dialogue_id] = {}
        persona_chat[dialogue_id]['persona2_ext'] = \
            induced_and_retrieved[dialogue_id]['induced_partner_personas']
        keys_to_keep = [
            'persona1_ori', 'persona1_ext', 'persona2_ext', 'text', 'text_plain', 'text_plain_cands'
        ]
        for k in keys_to_keep:
            persona_chat_extended_random_induced[dialogue_id][k] = persona_chat[dialogue_id][k]
    return persona_chat_extended_random_induced

def extend_with_retrieved_induced(persona_chat, induced_and_retrieved):
    persona_chat = copy.deepcopy(persona_chat)
    persona_chat_extended_retrieved_induced = {}
    for dialogue_id in persona_chat.keys():
        persona_chat_extended_retrieved_induced[dialogue_id] = {}
        persona_chat[dialogue_id]['persona1_ext'] = \
            induced_and_retrieved[dialogue_id]['relevant_speaker_personas']
        persona_chat[dialogue_id]['persona2_ext'] = \
            induced_and_retrieved[dialogue_id]['induced_partner_personas']
        keys_to_keep = [
            'persona1_ori', 'persona1_ext', 'persona2_ext', 'text', 'text_plain', 'text_plain_cands'
        ]
        for k in keys_to_keep:
            persona_chat_extended_retrieved_induced[dialogue_id][k] = persona_chat[dialogue_id][k]
    return persona_chat_extended_retrieved_induced


def main(mode):
    fp_pc_ext = f'{mode}_persona_original_chat_ext.json'
    persona_chat_ext = json.load(open(os.path.join(DATA_DIR, fp_pc_ext)))

    fp_ir = f'induced_and_retrieved_personas_{mode}.json'
    induced_and_retrieved = json.load(
        open(os.path.join(INDUCED_RETRIEVED_DIR, fp_ir))
    )
    
    fp_ir_turns = f'induced_and_retrieved_personas_with_turns_{mode}.json'
    induced_and_retrieved_with_turns = json.load(
        open(os.path.join(INDUCED_RETRIEVED_DIR, fp_ir_turns))
    )

    # Retrieved
    persona_chat_extended_retrieved = extend_with_retrieved(
        persona_chat_ext, induced_and_retrieved
    )
    with open(
        os.path.join(SAVE_DIR, f'{mode}_persona_original_chat_ext_retrieved.json'), 'w'
        ) as f:
        json.dump(persona_chat_extended_retrieved, f, indent=2)
    
    # Induced
    persona_chat_extended_induced = extend_with_induced(
        persona_chat_ext, induced_and_retrieved_with_turns
    )
    with open(
        os.path.join(SAVE_DIR, f'{mode}_persona_original_chat_ext_induced.json'), 'w'
        ) as f:
        json.dump(persona_chat_extended_induced, f, indent=2)
    
    # Random speaker facts + Induced
    persona_chat_extended_random_induced = extend_with_random_induced(
        persona_chat_ext, induced_and_retrieved_with_turns
    )
    with open(
        os.path.join(SAVE_DIR, f'{mode}_persona_original_chat_ext_random_induced.json'), 'w'
        ) as f:
        json.dump(persona_chat_extended_random_induced, f, indent=2)

    # Retrieved + Induced
    persona_chat_extended_retrieved_induced = extend_with_retrieved_induced(
        persona_chat_ext, induced_and_retrieved_with_turns
    )
    with open(
        os.path.join(SAVE_DIR, f'{mode}_persona_original_chat_ext_retrieved_induced.json'), 'w'
        ) as f:
        json.dump(persona_chat_extended_retrieved_induced, f, indent=2)



if __name__ == '__main__':
    DATA_DIR = '../data/persona_peacok'
    INDUCED_RETRIEVED_DIR = 'pickled_stuff'
    SAVE_DIR = '../data/persona_peacok'

    for mode in ['valid', 'train']:
        main(mode)
