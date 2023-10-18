import os
import pickle as pkl
import json
import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer

map_relation = {
    "characteristic": "here is my character trait",
    "routine_habit": "here is what i regularly or consistently do",
    "goal_plan": "here is what i will do or achieve in the future",
    "experience": "here is what i did in the past",
    "relationship": "related to other people or social groups"
}

def full_persona_triplets_to_sentences(full_persona_json):
    full_persona_sentences = {}
    for dialogue_id, personas in full_persona_json.items():
        full_persona_sentences[dialogue_id] = {}
        for persona_id, persona_triplets in personas.items():
            sentences = []
            for t in persona_triplets:
                sentences.append(f"{t[0]}. {map_relation[t[1]]}: {t[2]}")
            full_persona_sentences[dialogue_id][persona_id] = sentences
    return full_persona_sentences

def embed_full_personas(
    full_persona_sentences,
    gpu_id=None,
    embedding_dim=768,
    model_id='all-mpnet-base-v2'
    ):
    model = SentenceTransformer(model_id)
    for dialogue_id, personas in tqdm.tqdm(full_persona_sentences.items()):
        dialogue_embeddings = {}
        for persona_id, persona_sentences in personas.items():
            if len(persona_sentences) > 0:
                embeddings = model.encode(
                    persona_sentences, convert_to_tensor=True, device=f'cuda:{gpu_id}'
                )    
                if embedding_dim < embeddings.size(1):
                    embeddings = embeddings[:, :embedding_dim]
                embeddings = embeddings.cpu().numpy()
            else:
                embeddings = np.array([])
            dialogue_embeddings[f"{persona_id}_embeddings"] = embeddings
        full_persona_sentences[dialogue_id].update(dialogue_embeddings)
    return full_persona_sentences

def _clean_relations_in_triplets(full_persona_json):
    for dialogue_id, personas in full_persona_json.items():
        for persona_id, persona_triplets in personas.items():
            for i, t in enumerate(persona_triplets):
                for relation in map_relation.keys():
                    if t[1].startswith(relation):
                        full_persona_json[dialogue_id][persona_id][i][1] = relation
    return full_persona_json


def main():
    print('Full persona triplets to sentences...')
    full_personas_json = json.load(open(FULL_PERSONAS_PATH, 'r'))
    full_personas_json = _clean_relations_in_triplets(full_personas_json)
    full_persona_sentences = full_persona_triplets_to_sentences(full_personas_json)
    pkl.dump(
        full_persona_sentences,
        open(os.path.join(SAVE_DIR, 'full_persona_sentences.pkl'), 'wb')
    )
    print('Computing embeddings...')
    full_persona_embeddings = embed_full_personas(full_persona_sentences, gpu_id=GPU)
    pkl.dump(
        full_persona_embeddings,
        open(os.path.join(SAVE_DIR, 'full_persona_embeddings.pkl'), 'wb')
    )


if __name__ == '__main__':
    FULL_PERSONAS_PATH = '../data/persona_peacok/persona_extend_full_original.json'
    GPU = 7
    SAVE_DIR = 'pickled_stuff'
    os.makedirs(SAVE_DIR, exist_ok=True)
    main()

