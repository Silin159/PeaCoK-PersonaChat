import os
import pickle as pkl
import tqdm
import json
from similarity_searcher import (
    prepare_data_for_searcher,
    prepare_induced_persona_for_searcher,
    SimilaritySearcher
)

def induce_and_retrieve_single_dialogue(
    partner_persona_data,
    partner_utterance_data,
    speaker_persona_data,
    drop_inf=False
    ):
    # Induce partner persona
    if len(partner_persona_data) > 0:
        partner_persona_inducer = SimilaritySearcher(partner_persona_data)
        induced_partner_personas = partner_persona_inducer.find(
            partner_utterance_data, text_key='utterance', drop_inf=drop_inf
        )
    else:
        induced_partner_personas = {'texts': [], 'embeddings': []}
    # Prepare induced persona for retrieving
    induced_persona_data = prepare_induced_persona_for_searcher(induced_partner_personas)
    # Retrieve relevant speaker persona
    if len(speaker_persona_data) > 0:
        speaker_persona_retriever = SimilaritySearcher(speaker_persona_data)
        relevant_speaker_personas = speaker_persona_retriever.find(
            induced_persona_data, text_key='persona', drop_inf=drop_inf
        )
    else:
        relevant_speaker_personas = {'texts': [], 'embeddings': []}
    
    return {
        'induced': induced_partner_personas['texts'],
        'retrieved': relevant_speaker_personas['texts']
    }

def induce_and_retrieve_with_turns(
    full_personas_embedded,
    utterances_embedded,
    ):
    induced_partner_personas_per_dialogue = {}
    relevant_speaker_personas_per_dialogue = {}
    for dialogue_id in tqdm.tqdm(utterances_embedded):
        induced_partner_personas_per_dialogue[dialogue_id] = {}
        relevant_speaker_personas_per_dialogue[dialogue_id] = {}
        for turn in range(len(utterances_embedded[dialogue_id]['persona2_utterances'])):
            speaker_persona_data, partner_persona_data, partner_utterance_data = \
                prepare_data_for_searcher(
                    full_personas_embedded, utterances_embedded,
                    dialogue_id, turn
                )
            induced_and_retrieved = induce_and_retrieve_single_dialogue(
                partner_persona_data, partner_utterance_data, speaker_persona_data,
                drop_inf=True
            )
            induced_partner_personas_per_dialogue[dialogue_id][turn] = \
                induced_and_retrieved['induced']
            relevant_speaker_personas_per_dialogue[dialogue_id][turn] = \
                induced_and_retrieved['retrieved']
    return induced_partner_personas_per_dialogue, relevant_speaker_personas_per_dialogue

def induce_and_retrieve(full_personas_embedded, utterances_embedded):
    induced_partner_personas_per_dialogue = {}
    relevant_speaker_personas_per_dialogue = {}
    for dialogue_id in tqdm.tqdm(utterances_embedded):
        # Prepare dialogue for inducing
        speaker_persona_data, partner_persona_data, partner_utterance_data = \
            prepare_data_for_searcher(full_personas_embedded, utterances_embedded, dialogue_id)

        induced_and_retrieved = induce_and_retrieve_single_dialogue(
            partner_persona_data, partner_utterance_data, speaker_persona_data
        )

        induced_partner_personas_per_dialogue[dialogue_id] = induced_and_retrieved['induced']
        relevant_speaker_personas_per_dialogue[dialogue_id] = induced_and_retrieved['retrieved']
    return induced_partner_personas_per_dialogue, relevant_speaker_personas_per_dialogue

def format_as_json(
    utterances_embedded,
    induced_partner_personas_per_dialogue,
    relevant_speaker_personas_per_dialogue
    ):
    as_json = {}
    for dialogue_id in utterances_embedded.keys():
        as_json[dialogue_id] = {}
        as_json[dialogue_id]['partner_utterances'] = \
            utterances_embedded[dialogue_id]['persona2_utterances']
        as_json[dialogue_id]['induced_partner_personas'] = \
            induced_partner_personas_per_dialogue[dialogue_id]
        as_json[dialogue_id]['relevant_speaker_personas'] = \
            relevant_speaker_personas_per_dialogue[dialogue_id]
    return as_json


def main(mode, with_turns):
    full_personas_embedded = pkl.load(
        open(os.path.join(SAVE_DIR, 'full_persona_embeddings.pkl'), 'rb')
    )
    utterances_embedded = pkl.load(
        open(os.path.join(SAVE_DIR, f'utterance_embeddings_{mode}.pkl'), 'rb')
    )

    func = induce_and_retrieve_with_turns if with_turns else induce_and_retrieve
    induced_partner_personas_per_dialogue, relevant_speaker_personas_per_dialogue = \
        func(full_personas_embedded, utterances_embedded)

    with_turns_str = '_with_turns' if with_turns else ''
    pkl.dump(
        induced_partner_personas_per_dialogue,
        open(os.path.join(SAVE_DIR, f'induced_partner_personas{with_turns_str}_{mode}.pkl'), 'wb')
    )
    pkl.dump(
        relevant_speaker_personas_per_dialogue,
        open(os.path.join(SAVE_DIR, f'relevant_speaker_personas{with_turns_str}_{mode}.pkl'), 'wb')
    )

    as_json = format_as_json(
        utterances_embedded,
        induced_partner_personas_per_dialogue,
        relevant_speaker_personas_per_dialogue
    )
    with open(
        os.path.join(SAVE_DIR, f'induced_and_retrieved_personas{with_turns_str}_{mode}.json'), 'w'
        ) as f:
        json.dump(as_json, f, indent=2)

if __name__ == '__main__':
    SAVE_DIR = 'pickled_stuff'
    os.makedirs(SAVE_DIR, exist_ok=True)

    for mode in ['valid', 'train']:
        for with_turns in [True, False]:
            main(mode, with_turns)


