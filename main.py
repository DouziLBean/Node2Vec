import argparse
import networkx as nx
from gensim.models import word2vec

import Node2Vec

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--graph_path", default=".\graph\karate.edgelist")
    
    parser.add_argument("--walk_iter", default=10, help="number of random walk per node")
    parser.add_argument("--walk_length", default=5, help="length of each random walk")
    parser.add_argument("--p", default=1, help="return probability")
    parser.add_argument("--q", default=1, help="in-out probability")
    
    
    parser.add_argument("--vector_size", default=128, help="vector size in word2vec")
    parser.add_argument("--window_size", default=5, help="window size in word2vec")
    
    parser.add_argument("--save_model_path", default=".\model\model.bin")
    parser.add_argument("--save_embedding_path", default=".\embedding\karate.emb")
    
    return parser.parse_args()

def read_graph(path):
    G = nx.read_edgelist(path=path, nodetype=int, create_using=nx.DiGraph())
    # G.to_undirected()
    for edge in G.edges:
        G[edge[0]][edge[1]]["weight"] = 1
    return G

def main(args):
    G = read_graph(args.graph_path)
    node2vec = Node2Vec.Node2Vec(
        G,
        args.walk_iter,
        args.walk_length,
        args.p,
        args.q,
    )
    node2vec.preprocessModifiedWeights()
    walk_list = node2vec.gen_walk_list()
    walk_list = sorted(walk_list)
    
    # save random walks
    try:
        with open(".\save\walk_list.randomWalk", "w") as file:
            for walk in walk_list:
                file.write(str(walk) + "\n")
    except FileNotFoundError:
        with open(".\save\walk_list.randomWalk", "x") as file: # x: 创建模式
            for walk in walk_list:
                file.write(str(walk) + "\n")
    
    # convert random walks to sentences
    walk_list = [list(map(str, walk)) for walk in walk_list]
    
    # word2vec
    model = word2vec.Word2Vec(
        sentences=walk_list, 
        vector_size=args.vector_size,
        window=args.window_size,
    )
    model.save(args.save_model_path)
    # In versions of gensim library prior to version 4.0.0, the save_word2vec_format method was available in the Word2Vec class. 
    # However, starting from version 4.0.0, the method has been removed and replaced with a different approach for saving word vectors.
    # model.save_word2vec_format(args.save_embedding_path)
    # model = word2vec.Word2Vec.load_word2vec_format('\tmp\vectors.bin', binary=True)
    
    # model = word2vec.Word2Vec.load(args.save_model_path)
    node_vector_list = model.wv
    
    # save node embeddings
    try:
        with open(".\embedding\\node_embeddings.emb", "w") as file:
            for node in G.nodes:
                file.write(str(node_vector_list[str(node)]) + "\n")
    except FileNotFoundError:
        with open(".\embedding\\node_embeddings.emb", "x") as file:
            for node in G.nodes:
                file.write(str(node_vector_list[str(node)]) + "\n")
 
    
if __name__ == "__main__":
    args = parse_args()
    main(args)