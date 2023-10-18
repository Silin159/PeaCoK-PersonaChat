import os
import pickle as pkl
import json
import networkx as nx
from collections import OrderedDict
from sentence_transformers import SentenceTransformer

def json_to_nx_graph(peacok_json):
    valid_relations = ['routine_habit', 'characteristic', 'goal_plan', 'experience']
    invalid_distinctiveness = []
    
    g = nx.Graph()
    for persona, persona_to_attribute_info in peacok_json.items():
        persona = _first_to_third_person(persona)
        
        for attribute, attribute_info in persona_to_attribute_info['phrases'].items():
            relation = attribute_info['final']['majority'][0]
            distinctiveness = attribute_info['final']['majority'][2]
            if relation not in valid_relations or distinctiveness in invalid_distinctiveness:
                continue
            relation_informative = attribute_info['source_relation']

            g.add_node(persona, node_type='persona')
            g.add_node(attribute, node_type='attribute')
            g.add_edge(persona, attribute, relation=relation, relation_informative=relation_informative)
    return g

def graph_edges_to_natural_language(g):
    map_relation = {
        "characteristic": "here is his character trait",
        "routine_habit": "here is what he regularly or consistently does",
        "goal_plan": "here is what he will do or achieve in the future",
        "experience": "here is what he did in the past",
    }

    edges_nl = {}
    for e in g.edges():
        src, dst = e
        relation = g.edges()[e]['relation']
        if g.nodes()[src]['node_type'] == 'attribute':
            src, dst = dst, src
        edges_nl[e] = f'{src} {map_relation[relation]}: {dst}'

    return edges_nl

def embed_sentences(
    sentences,
    gpu_id=None,
    embedding_dim=768,
    model_id='all-mpnet-base-v2'
    ):
    # Load the SentenceTransformer model
    model = SentenceTransformer(model_id)
    # Embed the sentences
    embeddings = model.encode(sentences, convert_to_tensor=True, device=f'cuda:{gpu_id}')
    # Reduce the embedding dimensions if specified
    if embedding_dim < embeddings.size(1):
        embeddings = embeddings[:, :embedding_dim]
    # Return the embeddings
    return embeddings.cpu().numpy()

def _first_to_third_person(persona):
    replace_with = OrderedDict({'I am': 'He is', 'I work': 'He works', 'I': 'He'})
    for first_person, third_person in replace_with.items():
        if persona.startswith(first_person):
            return persona.replace(replace_with(first_person))
    return persona

def _print_graph_stats(g):
    print("#nodes, #edges: ", len(g.nodes()), len(g.edges()))

    print("Node type distribution:")
    for desired_node_type in ['persona', 'attribute']:
        count = sum(1 for node, attr in g.nodes(data=True) if attr['node_type'] == desired_node_type)
        print(desired_node_type, count)

    edge_types = {}
    for u, v, attr in g.edges(data=True):
        edge_type = attr['relation']
        if edge_type in edge_types:
            edge_types[edge_type] += 1
        else:
            edge_types[edge_type] = 1
    # Print the distribution of edge types
    print("Edge type distribution:")
    for edge_type, count in edge_types.items():
        print(f"{edge_type}: {count}")

    print("Number of connected components: ", nx.number_connected_components(g))


def main():
    peacok_json = json.load(open(PEACOK_PATH, 'r'))
    
    print('JSON to NX graph')
    g = json_to_nx_graph(peacok_json)
    _print_graph_stats(g)
    pkl.dump(g, open(os.path.join(SAVE_DIR, 'peacok_nx.pkl'), 'wb'))

    print('Graph edges to NL...')
    edges_nl = graph_edges_to_natural_language(g)
    pkl.dump(edges_nl, open(os.path.join(SAVE_DIR, 'peacok_edges_nl.pkl'), 'wb'))

    print(f'Computing sentence embeddings for {len(edges_nl)} sentences...')
    embeddings = embed_sentences(list(edges_nl.values()), gpu_id=GPU)
    edges_embeddings = {}
    for edge_id, embedding in zip(list(edges_nl.keys()), embeddings):
        edges_embeddings[edge_id] = embedding
    pkl.dump(edges_embeddings, open(os.path.join(SAVE_DIR, 'peacok_edges_embeddings.pkl'), 'wb'))


if __name__ == '__main__':
    PEACOK_PATH = '../data/peacok_kg/atomic_simple_head.json'
    GPU = 5
    SAVE_DIR = 'pickled_stuff'
    os.makedirs(SAVE_DIR, exist_ok=True)
    main()

