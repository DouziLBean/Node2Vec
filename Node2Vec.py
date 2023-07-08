import numpy as np
import random
import networkx as nx

class Node2Vec():
    def __init__(self, G, walk_iter, walk_length, p, q) -> None:
        self.G = G,
        self.G = self.G[0]
        self.walk_iter = walk_iter
        self.walk_length = walk_length
        self.p = p
        self.q = q
    
    def preprocessModifiedWeights(self) -> None:
        G = self.G
        p = self.p
        q = self.q
        
        # first step probs of random walk
        first_choice_prob_dict = {}
        for node in sorted(G.nodes):
            first_choice_prob_list = []
            for nbr in sorted(G.neighbors(node)):
                first_choice_prob_list.append(G[node][nbr]["weight"])
            first_choice_prob_list = [prob / sum(first_choice_prob_list) for prob in first_choice_prob_list]
            first_choice_prob_dict[node] = first_choice_prob_list
        
        # later steps probs of random walk based on a specific chosen edge
        next_choice_prob_dict = {}
        for edge in sorted(G.edges):
            pre = edge[0]
            cur = edge[1]
            next_choice_prob_list = []
            for nbr in sorted(G.neighbors(cur)):
                if nbr == pre:
                    next_choice_prob_list.append(G[cur][nbr]["weight"] / p)
                elif nbr in G.neighbors(pre):
                    next_choice_prob_list.append(G[cur][nbr]["weight"])
                else:
                    next_choice_prob_list.append(G[cur][nbr]["weight"] / q)
            next_choice_prob_list = [prob / sum(next_choice_prob_list) for prob in next_choice_prob_list]
            next_choice_prob_dict[edge] = next_choice_prob_list
            
        self.first_choice_prob_dict = first_choice_prob_dict
        self.next_choice_prob_dict = next_choice_prob_dict
    
    
    def gen_walk_list(self) -> list:
        walk_iter = self.walk_iter
        G = self.G
        
        walk_list = []
        for iter in range(walk_iter):
            for node in sorted(G.nodes):
                walk = self.gen_walk(node)
                walk_list.append(walk)
    
        return walk_list
                
                
    def gen_walk(self, src) -> list:
        G = self.G
        walk_length = self.walk_length
        first_choice_prob_dict = self.first_choice_prob_dict
        next_choice_prob_dict = self.next_choice_prob_dict
        
        walk = [src]
        
        # first step
        first_choice_prob_list = first_choice_prob_dict[src]
        first_choice_nbrID_list = sorted(G.neighbors(src))
        walk.append(self.alias_sampling(first_choice_prob_list, first_choice_nbrID_list))
        if walk[-1] == -1:
            walk.pop()
            return walk
        
        # later steps
        for l in range(2, walk_length):
            cur = walk[-1]
            pre = walk[-2]
            edge = (pre, cur)
            next_choice_prob_list = next_choice_prob_dict[edge]
            next_choice_nbrID_list = sorted(G.neighbors(cur))
            walk.append(self.alias_sampling(next_choice_prob_list, next_choice_nbrID_list))
            if walk[-1] == -1:
                walk.pop()
                break
        
        return walk


    def alias_sampling(self, prob_list_org, id_list_org) -> int:
        k = len(prob_list_org)
        if k == 0:
            return -1
        
        smaller = []
        larger = []
        
        id_list = np.zeros(k, dtype=int)
        prob_list = np.zeros(k, dtype=float)
        
        for i, prob in enumerate(prob_list_org):
            prob = prob * k
            prob_list[i] = prob
            if prob < 1:
                smaller.append(i)
            else:
                larger.append(i)
        
        while len(smaller) != 0 and len(larger) != 0:
            small = smaller.pop()
            large = larger.pop()
            
            id_list[small] = large
            
            prob = prob_list[large] - (1 - prob_list[small])
            if prob < 1:
                smaller.append(large)
            else:
                larger.append(large)
            prob_list[large] = prob
        
        # draw alias random sample
        id = random.randint(0, k - 1)
        if (random.random() < prob_list[id]):
            return id_list_org[id]
        else:
            return id_list_org[id_list[id]]
        
        
        