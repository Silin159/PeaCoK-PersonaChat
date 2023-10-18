import os
import pickle as pkl
import json
import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer

def prepare_utterances(persona_chat):
    utterances_per_dialogue = {}
    for dialogue_id, dialogue in persona_chat.items():
        utterances_per_dialogue[dialogue_id] = {}
        utterances = dialogue['text']
        utterances_per_dialogue[dialogue_id]['persona1_utterances'] = utterances[1::2]
        utterances_per_dialogue[dialogue_id]['persona2_utterances'] = utterances[0::2]
    return utterances_per_dialogue

def embed_utterances(
    utterances_per_dialogue,
    gpu_id=None,
    embedding_dim=768,
    model_id='all-mpnet-base-v2'
    ):
    model = SentenceTransformer(model_id)
    for dialogue_id, utterances_per_persona in tqdm.tqdm(utterances_per_dialogue.items()):
        dialogue_embeddings = {}
        for persona_id, utterances in utterances_per_persona.items():
            if len(utterances) > 0:
                embeddings = model.encode(
                    utterances, convert_to_tensor=True, device=f'cuda:{gpu_id}'
                )    
                if embedding_dim < embeddings.size(1):
                    embeddings = embeddings[:, :embedding_dim]
                embeddings = embeddings.cpu().numpy()
            else:
                embeddings = np.array([])
            dialogue_embeddings[f"{persona_id}_embeddings"] = embeddings
        utterances_per_dialogue[dialogue_id].update(dialogue_embeddings)
    return utterances_per_dialogue

def main(persona_chat_path, mode):
    print('Preparing PersonaChat utterances...')
    persona_chat = json.load(open(persona_chat_path, 'r'))
    utterances_per_dialogue = prepare_utterances(persona_chat)

    print('Computing embeddings...')
    utterance_embeddings = embed_utterances(utterances_per_dialogue, gpu_id=GPU)
    pkl.dump(
        utterance_embeddings,
        open(os.path.join(SAVE_DIR, f'utterance_embeddings_{mode}.pkl'), 'wb')
    )


if __name__ == '__main__':
    GPU = 7
    SAVE_DIR = 'pickled_stuff'
    os.makedirs(SAVE_DIR, exist_ok=True)

    main(
        '../data/persona_peacok/valid_persona_original_chat_convai2.json',
        'valid'
    )
    main(
        '../data/persona_peacok/train_persona_original_chat_convai2.json',
        'train'
    )
