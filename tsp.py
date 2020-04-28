# @file tsp.py
# @brief solve the traveling salesman problem
"""
minimize the travel cost for visiting n customers exactly once
approach:
    - start with assignment model
    - add cuts until there are no sub-cycles
    - two cutting plane possibilities (called inside "solve_tsp"):
        - addcut: limit the number of edges in a connected component S to |S|-1
        - addcut2: require the number of edges between two connected component to be >= 2

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""

import gzip
import math
import os
import pickle
import random

import networkx as nx
import numpy as np
from joblib import Parallel, delayed

import utilities
from pyscipopt import Model, quicksum, scip


def tsp2lp(V, c, filename):
    def addcut(cut_edges):

        # Initialize graph
        G = nx.Graph()
        G.add_edges_from(cut_edges)

        if nx.number_connected_components(G) == 1:
            return False

        model.freeTransform()

        for S in nx.connected_components(G):
            model.addCons(
                quicksum(x[i, j] for i in S for j in S if j > i) <= len(S) - 1
            )
            # print("cut: len(%s) <= %s" % (S, len(S) - 1))
        return True

    def addcut2(cut_edges):

        # Initialize graph
        G = nx.Graph()
        G.add_edges_from(cut_edges)

        if nx.number_connected_components(G) == 1:
            return False

        model.freeTransform()

        for S in nx.connected_components(G):
            T = set(V) - set(S)
            model.addCons(
                quicksum(x[i, j] for i in S for j in T if j > i)
                + quicksum(x[i, j] for i in T for j in S if j > i)
                >= 2
            )
        return True

    model = Model()
    model.hideOutput()  # silent/verbose mode
    x = {}
    for i in V:
        for j in V:
            if j > i:
                x[i,j] = model.addVar(ub=1, name="x(%s,%s)"%(i,j))

    for i in V:
        model.addCons(
            quicksum(x[j, i] for j in V if j < i)
            + quicksum(x[i, j] for j in V if j > i)
            == 2,
            "Degree(%s)" % i,
        )

    model.setObjective(
        quicksum(c[i, j] * x[i, j] for i in V for j in V if j > i), "minimize"
    )

    EPS = 1.e-6
    isMIP = False
    while True:
        model.optimize()
        edges = []
        for (i,j) in x:
            if model.getVal(x[i,j]) > EPS:
                edges.append( (i,j) )

        if addcut(edges) == False:
            if isMIP:     # integer variables, components connected: solution found
                break
            model.freeTransform()
            for (i,j) in x:     # all components connected, switch to integer model
                model.chgVarType(x[i,j], "B")
                isMIP = True

    model.writeProblem(filename)

    return model


def distance(x1, y1, x2, y2):
    """distance: euclidean distance between (x1,y1) and (x2,y2)"""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def make_problem(n):
    """make_problem: compute matrix distance based on euclidean distance"""
    V = range(1, n + 1)
    x = dict([(i, random.random()) for i in V])
    y = dict([(i, random.random()) for i in V])
    c = {}
    for i in V:
        for j in V:
            if j > i:
                c[i, j] = distance(x[i], y[i], x[j], y[j])
    return V, c


def main(n, filename):
    """
    Given the size of the vertex set |V|=n, generate and solve a random
    TSP instance.

    Parameters
    ----------
    n : int 
        Size of the vertex set.

    Returns
    -------
    V : vertex set
    c : list of edge costs
    obj : some model parameter
    edges : optimal solution
    """
    V, c = make_problem(n)
    tsp2lp(V, c, filename)


if __name__ == "__main__":
    import sys

    # Parse argument
    print("Usage: %s instance" % sys.argv[0])
    print("Using randomized example instead")
    seed = 1
    random.seed(seed)
    
    filenames = []
    
    n_nodes = 100
    n_samples = 10000
    os.makedirs(f'data/instances/tsp/train_{n_nodes}')
    filenames = [f'data/instances/tsp/train_{n_nodes}/instance_{i+1}.lp' for i in range(n_samples)]
    out = Parallel(n_jobs=-1)(delayed(main)(n_nodes, filenames[i]) for i in range(n_samples))

    n_nodes = 100
    n_samples = 2000
    os.makedirs(f'data/instances/tsp/test_{n_nodes}')
    filenames = [f'data/instances/tsp/test_{n_nodes}/instance_{i+1}.lp' for i in range(n_samples)]
    out = Parallel(n_jobs=-1)(delayed(main)(n_nodes, filenames[i]) for i in range(n_samples))

    n_nodes = 100
    n_samples = 2000
    os.makedirs(f'data/instances/tsp/valid_{n_nodes}')
    filenames = [f'data/instances/tsp/valid_{n_nodes}/instance_{i+1}.lp' for i in range(n_samples)]
    out = Parallel(n_jobs=-1)(delayed(main)(n_nodes, filenames[i]) for i in range(n_samples))

    n_nodes = 100
    n_samples = 100
    os.makedirs(f'data/instances/tsp/transfer_{n_nodes}')
    filenames = [f'data/instances/tsp/transfer_{n_nodes}/instance_{i+1}.lp' for i in range(n_samples)]
    out = Parallel(n_jobs=-1)(delayed(main)(n_nodes, filenames[i]) for i in range(n_samples))

    n_nodes = 150
    n_samples = 100
    os.makedirs(f'data/instances/tsp/transfer_{n_nodes}')
    filenames = [f'data/instances/tsp/transfer_{n_nodes}/instance_{i+1}.lp' for i in range(n_samples)]
    out = Parallel(n_jobs=-1)(delayed(main)(n_nodes, filenames[i]) for i in range(n_samples))
    
    n_nodes = 200
    n_samples = 100
    os.makedirs(f'data/instances/tsp/transfer_{n_nodes}')
    filenames = [f'data/instances/tsp/transfer_{n_nodes}/instance_{i+1}.lp' for i in range(n_samples)] 
    out = Parallel(n_jobs=-1)(delayed(main)(n_nodes, filenames[i]) for i in range(n_samples))
