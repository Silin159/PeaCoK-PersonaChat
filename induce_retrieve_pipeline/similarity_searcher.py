import random
import numpy as np
import faiss

class SimilaritySearcher:
    def __init__(
        self,
        index_data,
        text_key='persona',
        embedding_key='embedding'
        ):
        """
        index_data : list({text_key: ..., embedding_key: ...})

        If index_data is empty, the find method will return [] and np.array([]).
        """
        self.index_texts = [p[text_key] for p in index_data]
        self.index_embeddings = np.array(
            [p[embedding_key] for p in index_data]
        )
        # Create the FAISS index
        self.index = faiss.IndexFlatL2(self.index_embeddings.shape[1])
        self.index.add(self.index_embeddings)

    def find(
        self,
        query_data,
        k_final=5,
        k_per_query=1,
        text_key='persona',
        embedding_key='embedding',
        drop_inf=False
        ):
        """
        query_data : list({text_key: ..., embedding_key: ...})

        If query_data is empty, returns random k_final items from the index.
        """
        self.k_final = k_final
        self.k_per_query = k_per_query
        self.query_texts = [p[text_key] for p in query_data]
        self.query_embeddings = np.array(
            [p[embedding_key] for p in query_data]
        )

        if len(self.index_texts) == 0:
            return {'texts': [], 'embeddings': np.array([])}
        if len(self.query_texts) == 0:
            k = min(k_final, len(self.index_texts))
            random_choices = random.sample(range(len(self.index_texts)), k)
            return {
                'texts': [self.index_texts[i] for i in random_choices],
                'embeddings': self.index_embeddings[random_choices]
            }

        # Retrieve most similar index entries for each query
        scores, indices = self.index.search(
            self.query_embeddings, self.k_per_query
        )
        scores_per_index = self._collect_all_scores(indices, scores)
        self.sorted_best_scores = self._rank_indices_by_scores(
            scores_per_index, drop_inf
        )
        self.sorted_texts = [
            self.index_texts[index] for index in self.sorted_best_scores.keys()
        ]
        self.sorted_embeddings = [
            self.index_embeddings[index] for index in self.sorted_best_scores.keys()
        ]
        return {
            'texts': self.sorted_texts[:k_final],
            'embeddings': self.sorted_embeddings[:k_final]
        }

    def _collect_all_scores(self, indices, scores):
        num_full_persona_facts = len(self.index_texts)
        result_dict = {i: [] for i in range(num_full_persona_facts)}
        num_queries, k = indices.shape
        for i in range(num_queries):
            for j in range(k):
                index = indices[i, j]
                if index == -1:
                    # Means that there might be fewer indexed items than k
                    continue
                score = scores[i, j]
                result_dict[index].append(score)
        return result_dict

    def _rank_indices_by_scores(self, scores_per_index, drop_inf=False):
        best_score_per_index = {}
        # Keep only most similar
        for index, scores in scores_per_index.items():
            if len(scores) == 0:
                if drop_inf:
                    continue
                best_score_per_index[index] = float('inf')
            else:
                best_score_per_index[index] = min(scores)
        best_score_per_index = dict(
            sorted(best_score_per_index.items(), key=lambda item: item[1])
        )
        return best_score_per_index


def prepare_data_for_searcher(
    full_personas_embedded,
    utterances_embedded,
    dialogue_id,
    turn=None,
    ):
    speaker_persona_data = []
    for persona, embedding in zip(
        full_personas_embedded[dialogue_id]['persona1'],
        full_personas_embedded[dialogue_id]['persona1_embeddings']
        ):
        speaker_persona_data.append({'persona': persona, 'embedding': embedding})

    partner_persona_data = []
    for persona, embedding in zip(
        full_personas_embedded[dialogue_id]['persona2'],
        full_personas_embedded[dialogue_id]['persona2_embeddings']
        ):
        partner_persona_data.append({'persona': persona, 'embedding': embedding})

    partner_utterance_data = []
    indices_to_keep = (
        list(range(len(utterances_embedded[dialogue_id]['persona2_utterances'])))
        if turn is None else list(range(0, turn + 1))
    )
    utterances_to_keep = [
        utterances_embedded[dialogue_id]['persona2_utterances'][i] for i in indices_to_keep
    ]
    for utterance, embedding in zip(
        utterances_to_keep,
        utterances_embedded[dialogue_id]['persona2_utterances_embeddings'][indices_to_keep]
        ):
        partner_utterance_data.append({'utterance': utterance, 'embedding': embedding})

    return speaker_persona_data, partner_persona_data, partner_utterance_data

def prepare_induced_persona_for_searcher(induced_persona_embedded):
    induced_persona_data = []
    for persona, embedding in zip(
        induced_persona_embedded['texts'],
        induced_persona_embedded['embeddings']
        ):
        induced_persona_data.append({'persona': persona, 'embedding': embedding})
    return induced_persona_data
