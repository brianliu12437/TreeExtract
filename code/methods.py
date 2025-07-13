
import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum
import pickle

import numpy as np
import matplotlib.pyplot as plt

from itertools import product

from sklearn.model_selection import KFold

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
import sklearn
from numba import jit
import itertools
import scipy
import time
import gc
import copy
import random    
from sklearn.ensemble import BaggingRegressor
from scipy.sparse import csc_matrix
import sys


#### Plotting Functions
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def plot_tree(estimator, z, delete_rest =False, prog="dot", node_size=50):
    tree = estimator.tree_
    node_count = tree.node_count
    
    # 1. Basic checks
    if len(z) != node_count:
        raise ValueError(f"Length of z ({len(z)}) must match number of nodes ({node_count}).")

    children_left = tree.children_left
    children_right = tree.children_right

    # 2. Build a directed graph of all nodes
    G = nx.DiGraph()
    G.add_nodes_from(range(node_count))
    
    # Add edges
    for node_id in range(node_count):
        left_id = children_left[node_id]
        if left_id != -1:
            G.add_edge(node_id, left_id)
        
        right_id = children_right[node_id]
        if right_id != -1:
            G.add_edge(node_id, right_id)

    # 3. If delete=True, prune nodes not on a path to any highlighted node
    if delete_rest:
        # 3a. Build a parent array so we can climb from any node up to the root
        parent = np.full(node_count, -1, dtype=int)
        
        # Root is node 0. We'll fill parent[] by traversing from the root
        stack = [0]
        while stack:
            pid = stack.pop()
            lch = children_left[pid]
            if lch != -1:
                parent[lch] = pid
                stack.append(lch)
            rch = children_right[pid]
            if rch != -1:
                parent[rch] = pid
                stack.append(rch)
        
        # 3b. Find all nodes to keep
        keep_nodes = set()
        
        for node_id in range(node_count):
            if z[node_id] == 1:
                # Walk up the tree, adding ancestors
                cur = node_id
                while cur != -1 and cur not in keep_nodes:
                    keep_nodes.add(cur)
                    cur = parent[cur]
        
        # Subgraph with only the kept nodes
        G = G.subgraph(keep_nodes).copy()

    # 4. Use graphviz_layout (if available) for a tree-like arrangement
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog=prog)
    except ImportError:
        print("Warning: graphviz_layout not available, using spring_layout.")
        pos = nx.spring_layout(G)
    
    # 5. Create color map: red if z[node_id] = 1, else lightblue
    #    But only for nodes in the subgraph G.
    color_map = []
    for node_id in G.nodes():
        color_map.append("red" if z[node_id] == 1 else "lightblue")
    
    # 6. Draw
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=color_map)
    nx.draw_networkx_edges(G, pos, arrows=True)
    plt.axis("off")
    plt.show()




    
def get_nodes_to_keep(tree, z_array):
    """
    Identify nodes to keep based on z_array and their parent nodes.
    
    Parameters:
    - tree: DecisionTreeRegressor.tree_ object.
    - z_array: Boolean array indicating which nodes should be kept.

    Returns:
    - nodes_to_keep: Set of nodes that should be kept in the plot.
    - forced_leaf_nodes: Set of internal nodes that become leaf nodes due to z_array.
    """
    nodes_to_keep = set()
    forced_leaf_nodes = set()
    
    children_left = tree.children_left
    children_right = tree.children_right

    # Include all marked nodes and trace back to their parents
    for node_id in range(tree.node_count):
        if z_array[node_id]:  # If node is marked to keep
            nodes_to_keep.add(node_id)
            if children_left[node_id] != -1 or children_right[node_id] != -1:
                forced_leaf_nodes.add(node_id)  # Mark internal node as a forced leaf
            
            while node_id != -1:
                nodes_to_keep.add(node_id)
                node_id = np.where((children_left == node_id) | (children_right == node_id))[0]
                if len(node_id) == 0:
                    break
                node_id = node_id[0]  # Move to parent

    return nodes_to_keep, forced_leaf_nodes

def plot_ensemble_trees(tree_list, feature_names, z_array_list, prog="dot", node_size=50):
    """
    Plots an ensemble of decision trees in a **single shared plot**,
    keeping only the nodes indicated in z_array and their parent nodes.

    Parameters:
    - tree_list: List of decision tree estimators from an ensemble.
    - feature_names: List of feature names corresponding to input data.
    - z_array_list: List of boolean arrays, where each array corresponds to the nodes to keep for a tree.
    - prog: Graphviz layout program for tree visualization.
    - node_size: Size of nodes in the visualization.
    """
    
    # Color mapping for features (consistent across all trees)
    feature_colors = plt.cm.get_cmap("tab20c", len(feature_names))  # Generate distinct colors
    feature_to_color = {feature: feature_colors(i) for i, feature in enumerate(feature_names)}

    # Track features actually used in the plotted trees
    used_features = set()

    # Filter trees that have at least one selected node
    selected_trees = []
    selected_z_arrays = []
    for tree_idx, estimator in enumerate(tree_list):
        nodes_to_keep, forced_leaf_nodes = get_nodes_to_keep(estimator.tree_, z_array_list[tree_idx])
        if nodes_to_keep:  # Only keep trees with selected nodes
            selected_trees.append((estimator, nodes_to_keep, forced_leaf_nodes, z_array_list[tree_idx]))

    num_trees = len(selected_trees)
    if num_trees == 0:
        print("No trees have selected nodes to plot.")
        return

    # Create a single figure
    fig, ax = plt.subplots(figsize=(num_trees * 1.8, 3))  # Adjust width dynamically for a tighter fit

    # Calculate **even** spacing between trees
    total_width = num_trees * 3.5  # Adjusted spacing factor
    x_offset = -total_width / 2  # Start from the left and move right

    for idx, (tree, nodes_to_keep, forced_leaf_nodes, z_array) in enumerate(selected_trees):
        tree_structure = tree.tree_
        children_left = tree_structure.children_left
        children_right = tree_structure.children_right
        feature_indices = tree_structure.feature  # Feature indices for splits

        # Identify which features are actually used in the remaining nodes
        used_features.update(feature_names[feature_indices[n]] for n in nodes_to_keep if feature_indices[n] >= 0)

        # Build directed graph
        G = nx.DiGraph()
        for node_id in nodes_to_keep:
            G.add_node(node_id)

        # Assign colors based on features used for splitting
        color_map = []
        edge_colors = []
        for node_id in nodes_to_keep:
            if node_id in forced_leaf_nodes:
                color_map.append("white")  # Forced leaf nodes are empty
            elif feature_indices[node_id] >= 0:
                feature_name = feature_names[feature_indices[node_id]]
                color_map.append(feature_to_color[feature_name])
            else:
                color_map.append("none")  # Empty color for selected standard leaves
            
            edge_colors.append("black")  # Everything has black edges

        # Add edges
        for node_id in nodes_to_keep:
            left_id = children_left[node_id]
            right_id = children_right[node_id]
            if left_id in nodes_to_keep and node_id not in forced_leaf_nodes:
                G.add_edge(node_id, left_id)
            if right_id in nodes_to_keep and node_id not in forced_leaf_nodes:
                G.add_edge(node_id, right_id)

        # Compute layout
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog=prog)
        except ImportError:
            print("Warning: graphviz_layout not available, using spring_layout.")
            pos = nx.spring_layout(G)

        # **Fix spacing issues** by shifting trees evenly
        max_x = max(p[0] for p in pos.values()) if pos else 0
        pos = {node: (x + x_offset, y) for node, (x, y) in pos.items()}
        x_offset += max_x + 3  # Fine-tuned tree spacing

        # Plot the tree
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=color_map, edgecolors=edge_colors, linewidths=1.5)
        nx.draw_networkx_edges(G, pos, arrows=True)

    # Remove axis
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    # Create a legend **only for used features**, outside the plot
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=feature_to_color[feat], markersize=8, markeredgecolor="black")
                      for feat in used_features]
    
    fig.legend(
    legend_handles, 
    used_features, 
    title="Features Used", 
    loc="lower center",  # Moves legend to the bottom
    fontsize=15, 
    ncol=min(5, len(used_features)),  
    bbox_to_anchor=(0.5, -0.6)  # Moves legend **below** the plot
    )
    plt.show()


### Helper Functions
def count_nodes_after_pruning(tree, z):
    """
    Count the number of nodes remaining in a decision tree after pruning nodes 
    that are not on a path to any highlighted node (where z[node] = 1).

    Parameters:
    - tree: A trained sklearn DecisionTreeRegressor
    - z: A binary numpy array indicating which nodes should be treated as leaf nodes

    Returns:
    - remaining_node_count: The number of nodes remaining after pruning
    """
    tree_structure = tree.tree_
    node_count = tree_structure.node_count

    # Ensure z has correct length
    if len(z) != node_count:
        raise ValueError(f"Length of z ({len(z)}) must match number of nodes ({node_count}).")

    children_left = tree_structure.children_left
    children_right = tree_structure.children_right

    # Step 1: Build parent array to track node hierarchy
    parent = np.full(node_count, -1, dtype=int)
    
    stack = [0]  # Start from root
    while stack:
        pid = stack.pop()
        lch = children_left[pid]
        if lch != -1:
            parent[lch] = pid
            stack.append(lch)
        rch = children_right[pid]
        if rch != -1:
            parent[rch] = pid
            stack.append(rch)
    
    # Step 2: Identify nodes to keep (all ancestors of z-marked nodes)
    keep_nodes = set()

    for node_id in range(node_count):
        if z[node_id] == 1:
            cur = node_id
            while cur != -1 and cur not in keep_nodes:
                keep_nodes.add(cur)
                cur = parent[cur]

    # Step 3: Count remaining nodes
    remaining_node_count = len(keep_nodes)

    return remaining_node_count

def get_node_depths(tree):
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    node_count = tree.tree_.node_count
    depths = np.zeros(shape=node_count, dtype=int)
    stack = [(0, 0)]  # (node_id, depth)
    while stack:
        node_id, depth = stack.pop()
        depths[node_id] = depth
        # If this node has a left child, push it on stack with depth+1
        if children_left[node_id] != -1:
            stack.append((children_left[node_id], depth + 1))
        # If this node has a right child, push it on stack with depth+1
        if children_right[node_id] != -1:
            stack.append((children_right[node_id], depth + 1))
    return depths


def get_all_descendants(tree):
    """
    Return a dict mapping node_id -> list of all descendant node_ids
    (including direct children, grandchildren, etc.).
    """
    children_left = tree.children_left
    children_right = tree.children_right
    node_count = tree.node_count
    
    # We'll store the result in a dictionary
    all_descendants = [[] for _ in range(node_count)]
    
    def dfs(node_id, parent_id):
        """DFS from node_id, accumulating descendants in all_descendants[parent_id]."""
        if node_id == -1:
            return
        # Add this child to the parent's descendant list
        all_descendants[parent_id].append(node_id)
        
        # Recurse
        dfs(children_left[node_id], parent_id)
        dfs(children_right[node_id], parent_id)
    
    # For each node, do a DFS down both branches from that node
    for node_id in range(node_count):
        left = children_left[node_id]
        right = children_right[node_id]
        dfs(left, node_id)
        dfs(right, node_id)
    
    return {node_id: desc_list for node_id, desc_list in enumerate(all_descendants)}



def get_all_parents(tree):
    """
    Return a dict mapping node_id -> list of all ancestor (parent) node_ids.
    The ancestor list is in order [parent, grandparent, ..., root].
    
    Parameters
    ----------
    tree : sklearn.tree._tree.Tree
        The underlying tree object (e.g. model.tree_).
    
    Returns
    -------
    all_parents : dict
        A dictionary where all_parents[node_id] is a list of node_ids
        that are ancestors of 'node_id'. If node_id is the root (0),
        its ancestors list is empty [].
    """
    children_left = tree.children_left
    children_right = tree.children_right
    node_count = tree.node_count
    
    # Parent array: parent[node] = the direct parent of 'node' (or -1 if none)
    parent = [-1] * node_count

    # Root is always node 0. We'll do a simple DFS or BFS from the root
    stack = [0]
    while stack:
        node_id = stack.pop()
        
        left_child = children_left[node_id]
        if left_child != -1:
            parent[left_child] = node_id
            stack.append(left_child)
        
        right_child = children_right[node_id]
        if right_child != -1:
            parent[right_child] = node_id
            stack.append(right_child)

    # Now gather all ancestors for each node by climbing up the parent array
    all_parents = {}
    for node_id in range(node_count):
        ancestors = []
        current = node_id
        # climb up until there's no parent
        while parent[current] != -1:
            p = parent[current]
            ancestors.append(p)
            current = p
        
        all_parents[node_id] = ancestors
    
    return all_parents


def get_tree_details(tree1,X):
    num_nodes = tree1.tree_.node_count
    M = tree1.decision_path(X).tocsc()
    n_obs = tree1.tree_.n_node_samples
    depths = get_node_depths(tree1)
    c_array = list(get_all_descendants(tree1.tree_).values())
    p_array = list(get_all_parents(tree1.tree_).values())
    parents_children_array = [c_array[i] + p_array[i] for i in range(len(c_array))]
    
    return M, num_nodes, parents_children_array, c_array , p_array, n_obs, depths


from sklearn.tree import DecisionTreeRegressor

def get_parent_features_list(regressor):
    tree = regressor.tree_
    n_nodes = tree.node_count
    parent_features = [[] for _ in range(n_nodes)]

    def traverse(node, ancestors):
        parent_features[node] = ancestors.copy()
        if tree.children_left[node] != -1:
            feature = tree.feature[node]
            traverse(tree.children_left[node], ancestors + [feature])
            traverse(tree.children_right[node], ancestors + [feature])

    traverse(0, [])
    return parent_features

def get_node_feat(tree_list):
    return [[ list(np.unique(featset)) for featset in get_parent_features_list(tree)] for tree in tree_list]



### setup functions

def get_tree_details(tree1,X):
    num_nodes = tree1.tree_.node_count
    M = tree1.decision_path(X).tocsc()
    n_obs = tree1.tree_.n_node_samples
    depths = get_node_depths(tree1)
    c_array = list(get_all_descendants(tree1.tree_).values())
    p_array = list(get_all_parents(tree1.tree_).values())
    parents_children_array = [c_array[i] + p_array[i] for i in range(len(c_array))]
    
    return M, num_nodes, parents_children_array, c_array , p_array, n_obs, depths

### gets block indicies for tree structure
def block_indices(num_nodes_ensemble):
    blocks = []
    start_idx = 0
    for size in num_nodes_ensemble:
        # indices for this block range from start_idx to start_idx+size-1
        block = list(range(start_idx, start_idx + size))
        blocks.append(block)
        start_idx += size
    return blocks

def get_ensemble_details(X, tree_list):
    c_all = []
    depths_ensemble = []
    n_obs_ensemble = []
    n_nodes_ensemble = []
    M_all = []
    for tree1 in tree_list:
        M, num_nodes, parents_children_array, c_array , p_array, n_obs, depths = get_tree_details(tree1,X)
        M_all.append(M)
        n_nodes_ensemble.append(num_nodes)
        n_obs_ensemble.append(n_obs)
        depths_ensemble.append(depths+1)
        c_all.append(c_array)
    blocks = block_indices(n_nodes_ensemble)
    
    return M_all,c_all , depths_ensemble, n_obs_ensemble, n_nodes_ensemble,blocks
    

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

def get_pred(M_all, z_array, w_array):
    pred = np.zeros(M_all[0].shape[0])
    for i in range(len(M_all)):
        pred = pred + M_all[i]@w_array[i]
    return pred


def get_regression_cost_subgradient_sp(X, y, s, lambda_ridge, a, lambda_penalized):
    p = X.shape[1]
    X_sub = X[:, s == 1]  # Extract selected columns

    if X_sub.shape[1] > 0:  # Ensure at least one feature is selected
        # Compute (lambda_ridge * I + X_sub'X_sub)^(-1) * X_sub' * y
        I_lambda = sp.eye(X_sub.shape[1]) / lambda_ridge
        XtX = X_sub.T @ X_sub  # Sparse
        rhs = X_sub.T @ y  # Sparse-friendly multiplication
        alpha_star = y - X_sub @ spsolve(I_lambda + XtX, rhs)  # Efficient sparse solve
    else:
        alpha_star = y  # If no features are selected, just return y

    # Compute cost function
    c = 0.5 * (y.T @ alpha_star).item() + lambda_penalized * np.sum(a[s == 1])

    # Compute subgradients without converting sparse columns to dense
    subgradients = np.zeros(p)
    X_alpha = X.T @ alpha_star  # Efficient sparse multiplication

    for j in range(p):
        subgradients[j] = -0.5 * lambda_ridge * (X_alpha[j]) ** 2 + lambda_penalized * a[j]

    return c, alpha_star, subgradients




def solve_subproblem_sp(y, M,c_array, lambda_ridge, lambda_penalized, a ,tol , mipgap = 0.001):
    regressor = gp.Model()
    regressor.params.OutputFlag = 0
    regressor.params.timelimit = 600
    regressor.params.mipgap = mipgap 
    
    dim = M.shape[1]
    
    nu = regressor.addVar(name="nu") 
    #z_new = regressor.addVars(dim, vtype=GRB.BINARY, name="z_new") 
    z_new = regressor.addMVar(dim, vtype=GRB.BINARY, name="z_new") 
    z_new1 = z_new.tolist()
    regressor.setObjective(nu, GRB.MINIMIZE)
    regressor.addConstr(nu>=0)
    big_M = [len(p) for p in c_array]

    for i in range(dim):
        regressor.addConstr(gp.quicksum([z_new1[j] for j in c_array[i]]) <= (1 - z_new1[i])*big_M[i])
    

    t1 = time.time()
    cost = 10**9
    nu1 = 0
    n_solves = 0
    while  (nu1 - cost)/cost < -tol:
        regressor.optimize()
        n_solves = n_solves + 1
        z = np.array([z_new1[i].X for i in range(dim)])
        nu1 = nu.X
        z= z.astype(int)
        cost, alpha_star, subgradient = get_regression_cost_subgradient_sp(M,y,z,lambda_ridge,a,lambda_penalized)  
        #regressor.addConstr(nu >= cost + quicksum([ subgradient[i]*(z_new[i] -z[i]) for i in range(dim)]))
        regressor.addConstr(nu >= cost +subgradient@(z_new - z) )
        
        #print(nu1,cost)
        
    M_sub = M[:, z == 1]  # Extract relevant columns

 
    reg_value = 1.0 / lambda_ridge
    I_reg = sp.diags([reg_value] * M_sub.shape[1])

    XtX = M_sub.T @ M_sub 
    rhs = M_sub.T @ y  

    w_sub = spsolve(XtX + I_reg, rhs)
    w = np.zeros(len(z))
    w[z==1] = w_sub
    return w,z, n_solves


def CBCD_solve(y, M_all,c_all,blocks, lambda_ridge, lambda_penalized , a_array = [],
               warm_start = [], debias = True,  ILP_tol = 0.001, cbcd_tol = 10**-4):
    
    if len(warm_start) == 0:
        z_array = [np.zeros(len(temp)) for temp in blocks]
        w_array = [np.zeros(len(temp)) for temp in blocks]
        
    else:
        z_array = warm_start[0]
        w_array = warm_start[1]
        
        
    n_samples = M_all[0].shape[0]
    ntrees = len(M_all)

    pred = get_pred(M_all, z_array, w_array)
    
    if len(a_array) == 0:
        ridge_penalty =  0.5*(1/lambda_ridge)*np.sum([w@w for w in w_array])
        penalized_term = lambda_penalized*np.sum([sum(z) for z in z_array])
        obj = 0.5*(y-pred)@(y-pred) + ridge_penalty + penalized_term
    else:
        ridge_penalty =  0.5*(1/lambda_ridge)*np.sum([w@w for w in w_array])
        penalized_term = lambda_penalized*np.sum([np.sum(z * a) for z, a in zip(z_array, a_array)])
        obj  = 0.5*(y-pred)@(y-pred) + ridge_penalty + penalized_term


    loss = [obj]
    cycle_loss = [obj]
    converged = False
    
    n_solves_all = []
    while converged == False:
        for t1 in range(ntrees):
            if len(a_array) == 0:
                a = np.ones(M_all[t1].shape[1])
                penalized_term = penalized_term - lambda_penalized*np.sum(z_array[t1])
            else:
                a = a_array[t1]
                penalized_term = penalized_term - lambda_penalized*np.sum(z_array[t1]*a)
                
            pred = pred - M_all[t1]@w_array[t1]
            ridge_penalty = ridge_penalty -  0.5*(1/lambda_ridge)*np.sum(w_array[t1]@w_array[t1])
            

            w,z,n_solves = solve_subproblem_sp(y - pred,  M_all[t1],c_all[t1], 
                                               lambda_ridge, lambda_penalized,a ,ILP_tol)
            n_solves_all.append(n_solves)
            
            w_array[t1] = w
            z_array[t1] = z
            
            pred = pred + M_all[t1]@w_array[t1]
            ridge_penalty = ridge_penalty +  0.5*(1/lambda_ridge)*np.sum(w@w)
            
            if len(a_array) == 0:
                penalized_term = penalized_term + lambda_penalized*np.sum(z_array[t1])
            else:
                penalized_term = penalized_term + lambda_penalized*np.sum(z_array[t1]*a)
            
            obj = 0.5*(y-pred)@(y-pred) + ridge_penalty + penalized_term
            loss.append(obj)

        cycle_loss.append(obj)
        if len(cycle_loss) > 2:
            if (cycle_loss[-2] - cycle_loss[-1])/n_samples <= cbcd_tol:
                converged = True

    z_stacked = np.hstack([z111 for z11 in z_array for z111 in z11])
    
    if debias == True:
        M_sub_list = [M[:, z_sub == 1] for M, z_sub in zip(M_all, z_array)]
        M_sub = sp.hstack(M_sub_list, format='csc') 
        reg_term = sp.diags([0.001] * M_sub.shape[1])
        XtX = M_sub.T @ M_sub
        rhs = M_sub.T @ y 
        w_sub = spsolve(XtX + reg_term, rhs)  
        w_stacked = np.zeros(len(z_stacked)) 
        w_stacked[z_stacked == 1] = w_sub 
    else:
        w_stacked = np.hstack([w111 for w11 in w_array for w111 in w11])

    return w_stacked, z_stacked,w_array, z_array, cycle_loss, loss, n_solves_all


def CBCD_path(lambda_penalized_range, y, M_all,c_all,blocks,
        lambda_ridge , a_array = [],warm_start = True, debias = True,  ILP_tol = 0.001, cbcd_tol = 10**-3):

    ws = []
    w_res = []
    z_res = []
    solves = []
    losses = []
    for lambda_p in lambda_penalized_range:

        t1 = time.time()
        w_stacked1, z_stacked1, w_array1, z_array1, cycle_loss1, loss1,n_solves1 = CBCD_solve(y, M_all,c_all,blocks, 
                                                                                lambda_ridge, lambda_p , a_array = a_array,
                                           warm_start = ws, debias = debias,  ILP_tol = ILP_tol, cbcd_tol = cbcd_tol)

        print(time.time()- t1, lambda_p)
        
        if warm_start == True:
            ws = [copy.deepcopy(z_array1), copy.deepcopy(w_array1)]

        w_res.append(w_array1)
        z_res.append(z_array1)
        solves.append(n_solves1)
        losses.append(loss1)

    return w_res, z_res, solves, losses

from scipy.sparse.linalg import splu


def get_regression_cost_subgradient_preload(X, y, s, lambda_ridge, a, lambda_penalized,LU):
    p = X.shape[1]
    X_sub = X[:, s == 1]  # Extract selected columns
    alpha_star = y - X_sub @ LU.solve(X_sub.T @ y)
 
    # Compute cost function
    c = 0.5 * (y.T @ alpha_star).item() + lambda_penalized * np.sum(a[s == 1])

    # Compute subgradients without converting sparse columns to dense
    subgradients = np.zeros(p)
    X_alpha = X.T @ alpha_star  # Efficient sparse multiplication

    #for j in range(p):
     #   subgradients[j] = -0.5 * lambda_ridge * (X_alpha[j]) ** 2 + lambda_penalized * a[j]
    subgradients = -0.5 * lambda_ridge * (X_alpha ** 2) + lambda_penalized * a

    return c, alpha_star, subgradients

def get_regression_cost_subgradient_splu(X, y, s, lambda_ridge, a, lambda_penalized):
    p = X.shape[1]
    X_sub = X[:, s == 1]  # Extract selected columns

    if X_sub.shape[1] > 0:  # Ensure at least one feature is selected
        # Compute (lambda_ridge * I + X_sub'X_sub)^(-1) * X_sub' * y
        I_lambda = sp.eye(X_sub.shape[1]) / lambda_ridge
        XtX = X_sub.T @ X_sub  # Sparse
        rhs = X_sub.T @ y  # Sparse-friendly multiplication
        LU = splu(I_lambda + XtX)
        alpha_star = y - X_sub @ LU.solve(rhs) # Efficient sparse solve
    else:
        alpha_star = y  # If no features are selected, just return y
        LU = None

    # Compute cost function
    c = 0.5 * (y.T @ alpha_star).item() + lambda_penalized * np.sum(a[s == 1])

    # Compute subgradients without converting sparse columns to dense
    subgradients = np.zeros(p)
    X_alpha = X.T @ alpha_star  # Efficient sparse multiplication

    for j in range(p):
        subgradients[j] = -0.5 * lambda_ridge * (X_alpha[j]) ** 2 + lambda_penalized * a[j]

    return c, alpha_star, subgradients, LU



def solve_subproblem_splu_preload(y, M,c_array, lambda_ridge, lambda_penalized, a,
                            preload_tree, preload_LU ,tol , mipgap = 0.001):
    regressor = gp.Model()
    regressor.params.OutputFlag = 0
    regressor.params.timelimit = 600
    regressor.params.mipgap = mipgap 
    
    dim = M.shape[1]
    
    nu = regressor.addVar(name="nu") 
    z_new = regressor.addMVar(dim, vtype=GRB.BINARY, name="z_new") 
    z_new1 = z_new.tolist()
    regressor.setObjective(nu, GRB.MINIMIZE)
    regressor.addConstr(nu>=0)
    big_M = [len(p) for p in c_array]

    for i in range(dim):
        regressor.addLConstr(gp.quicksum([z_new1[j] for j in c_array[i]]) <= (1 - z_new1[i])*big_M[i])
        
    
    cost = 10**9
    nu1 = 0
    n_solves = 0
    
    #preload constraints
    if preload_tree.shape[0] > 0:
        for preload_row in range(preload_tree.shape[0]):
            z_preload = np.array(preload_tree.getrow(preload_row).toarray().ravel())
            if sum(z_preload) > 0:
                cost, alpha_star, subgradient = get_regression_cost_subgradient_preload(M,y,z_preload,
                                                lambda_ridge,a,lambda_penalized, preload_LU[preload_row])  
                
                
                
                #inner= [ subgradient[i]*(z_new[i] -z_preload[i]) for i in range(dim)]
            
       
                subgradient_z_preload = subgradient @ z_preload      # numeric
                expr = gp.LinExpr(subgradient, z_new1)               # Gurobi LinExpr
                regressor.addLConstr(nu >= cost + expr - subgradient_z_preload)
                
                #regressor.addConstr(nu >= cost +subgradient@(z_new - z_preload))
                #regressor.addLConstr(nu >= cost + quicksum([ subgradient[i]*(z_new[i] -z_preload[i]) for i in range(dim)]))

    
    z_all = sp.lil_matrix((0, dim), dtype=bool)  
    LU_all = []
    
    t1 = time.time()
   
    while  (nu1 - cost)/cost < -tol:
        regressor.optimize()
        n_solves = n_solves + 1
        z = np.array([z_new1[i].X for i in range(dim)])
        nu1 = nu.X
        z= z.astype(int)
        cost, alpha_star, subgradient, LU = get_regression_cost_subgradient_splu(M,y,z,lambda_ridge,a,lambda_penalized)  
        
        
        subgradient_z = subgradient @ z      # numeric
        expr = gp.LinExpr(subgradient, z_new1)               # Gurobi LinExpr
        regressor.addLConstr(nu >= cost + expr - subgradient_z)

        
        if sum(z) > 0:
            z_all.resize((z_all.shape[0] + 1, dim))  # Add new row
            z_all[-1, :] = z 
            LU_all.append(LU)
        #print(nu1,cost)
        
    M_sub = M[:, z == 1]  # Extract relevant columns

 
    reg_value = 1.0 / lambda_ridge
    I_reg = sp.diags([reg_value] * M_sub.shape[1])

    XtX = M_sub.T @ M_sub 
    rhs = M_sub.T @ y  

    w_sub = spsolve(XtX + I_reg, rhs)
    w = np.zeros(len(z))
    w[z==1] = w_sub
    z_all = z_all.tocsr()
    
    return w,z, n_solves, z_all, LU_all


def CBCD_solve_preloadLU(y, M_all,c_all, blocks, lambda_ridge, lambda_penalized , a_array = [],
               warm_start = [],preload_solns = [], preload_LU = [] , debias = True,  ILP_tol = 0.0001, cbcd_tol = 10**-4):
    
    if len(warm_start) == 0:
        z_array = [np.zeros(len(temp)) for temp in blocks]
        w_array = [np.zeros(len(temp)) for temp in blocks]
        
    else:
        z_array = warm_start[0]
        w_array = warm_start[1]
        
        
    n_samples = M_all[0].shape[0]
    ntrees = len(M_all)

    pred = get_pred(M_all, z_array, w_array)
    
    if len(a_array) == 0:
        ridge_penalty =  0.5*(1/lambda_ridge)*np.sum([w@w for w in w_array])
        penalized_term = lambda_penalized*np.sum([sum(z) for z in z_array])
        obj = 0.5*(y-pred)@(y-pred) + ridge_penalty + penalized_term
    else:
        ridge_penalty =  0.5*(1/lambda_ridge)*np.sum([w@w for w in w_array])
        penalized_term = lambda_penalized*np.sum([np.sum(z * a) for z, a in zip(z_array, a_array)])
        obj  = 0.5*(y-pred)@(y-pred) + ridge_penalty + penalized_term


    loss = [obj]
    cycle_loss = [obj]
    converged = False
    
    n_solves_all = []
    
    if len(preload_solns) == 0:
        preload_solns = [sp.csr_matrix((0, M_all[i].shape[1]), dtype=int) for i in range(len(M_all))]
        preload_LU = [[] for _ in range(len(M_all))]  
    else:
        preload_solns = preload_solns
        preload_LU = preload_LU
        
    while converged == False:
        for t1 in range(ntrees):
            
            if len(a_array) == 0:
                a = np.ones(M_all[t1].shape[1])
                penalized_term = penalized_term - lambda_penalized*np.sum(z_array[t1])
            else:
                a = a_array[t1]
                penalized_term = penalized_term - lambda_penalized*np.sum(z_array[t1]*a)
                
            pred = pred - M_all[t1]@w_array[t1]
            ridge_penalty = ridge_penalty -  0.5*(1/lambda_ridge)*np.sum(w_array[t1]@w_array[t1])
            

            w,z,n_solves, z_all, LU_all = solve_subproblem_splu_preload(y - pred,  M_all[t1],c_all[t1], 
                                               lambda_ridge, lambda_penalized,a , preload_solns[t1],preload_LU[t1], ILP_tol)
            n_solves_all.append(n_solves)
            
            
            preload_solns[t1] = sp.vstack([preload_solns[t1], z_all], format='csr')
            preload_LU[t1].extend(LU_all)
            
            #uncomment these lines for condensed version
            #preload_solns[t1] = z_all
            #preload_LU[t1] = LU_all

            w_array[t1] = w
            z_array[t1] = z
            
            pred = pred + M_all[t1]@w_array[t1]
            ridge_penalty = ridge_penalty +  0.5*(1/lambda_ridge)*np.sum(w@w)
            
            if len(a_array) == 0:
                penalized_term = penalized_term + lambda_penalized*np.sum(z_array[t1])
            else:
                penalized_term = penalized_term + lambda_penalized*np.sum(z_array[t1]*a)
            
            obj = 0.5*(y-pred)@(y-pred) + ridge_penalty + penalized_term
            loss.append(obj)

        cycle_loss.append(obj)
        if len(cycle_loss) > 2:
            if (cycle_loss[-2] - cycle_loss[-1])/n_samples <= cbcd_tol:
                converged = True

    z_stacked = np.hstack([z111 for z11 in z_array for z111 in z11])
    
    if debias == True:
        M_sub_list = [M[:, z_sub == 1] for M, z_sub in zip(M_all, z_array)]
        M_sub = sp.hstack(M_sub_list, format='csc') 
        reg_term = sp.diags([0.001] * M_sub.shape[1])
        XtX = M_sub.T @ M_sub
        rhs = M_sub.T @ y 
        w_sub = spsolve(XtX + reg_term, rhs)  
        w_stacked = np.zeros(len(z_stacked)) 
        w_stacked[z_stacked == 1] = w_sub 
    else:
        w_stacked = np.hstack([w111 for w11 in w_array for w111 in w11])

    return w_stacked, z_stacked,w_array, z_array, cycle_loss, loss, n_solves_all,preload_solns,preload_LU



def CBCD_path_preload(lambda_penalized_range, y, M_all,c_all,blocks,
        lambda_ridge , a_array = [],warm_start = True, debias = True,  ILP_tol = 0.001, cbcd_tol = 10**-3, early_stop_support = 0):

    ws = []
    w_res = []
    z_res = []
    solves = []
    losses = []
    preload_solns = []
    preload_LU = []
    for lambda_p in lambda_penalized_range:

        t1 = time.time()
        w_stacked1, z_stacked1, w_array1, z_array1, cycle_loss1,\
        loss1,n_solves1,preloadsolns1, preloadLU1 = CBCD_solve_preloadLU(y, M_all,c_all,blocks, 
                                    lambda_ridge, lambda_p , a_array = a_array,
                                    warm_start = ws,preload_solns = preload_solns, preload_LU = preload_LU
                                    ,  debias = debias,  ILP_tol = ILP_tol, cbcd_tol = cbcd_tol)

        print(time.time()- t1, lambda_p)
        
        if warm_start == True:
            ws = [copy.deepcopy(z_array1), copy.deepcopy(w_array1)]
            preload_solns = preloadsolns1
            preload_LU = preloadLU1
            
        w_res.append(w_array1)
        z_res.append(z_array1)
        solves.append(n_solves1)
        losses.append(loss1)
        if early_stop_support > 0:
            if (sum([sum(z) for z in z_array1])) > early_stop_support:
                 return w_res, z_res, solves, losses
        
    return w_res, z_res, solves, losses


##### Linear Relaxation

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
def get_integer_soln(z_frac, c_array,mipgap = 0.001):
    regressor = gp.Model()
    regressor.params.OutputFlag = 0
    regressor.params.timelimit = 600
    regressor.params.mipgap = mipgap 
    dim = len(z_frac)
    z_round = regressor.addMVar(dim, vtype=GRB.BINARY, name="z_round") 
    z_round1 = z_round.tolist()
    regressor.setObjective(z_frac@z_round, GRB.MAXIMIZE)
    big_M = [len(p) for p in c_array]

    for i in range(dim):
        regressor.addConstr(gp.quicksum([z_round1[j] for j in c_array[i]]) <= (1 - z_round1[i])*big_M[i])
    regressor.optimize()
    
    return np.array([z_round1[i].X for i in range(dim)])

def objective_and_gradient_sparse_csc(X, Y, s, gamma, tol=1e-12):
    """
    A faster version of the objective+gradient calculation when s is sparse,
    allowing X to be a scipy.sparse CSC matrix.

    Uses the Woodbury identity:
        (I + U U^T)^{-1} = I - U (I + U^T U)^{-1} U^T,
    with U built from the nonzero entries of s.

    Parameters
    ----------
    X : scipy.sparse.csc_matrix, shape (n, p)
        Matrix whose j-th column is x_j (an n-dimensional vector).
        Must be in CSC format for efficient getcol(j) access.
    Y : np.ndarray, shape (n,)
        The n-dimensional vector Y.
    s : np.ndarray, shape (p,)
        The sparse coefficient vector (many entries are zero).
    gamma : float
        Scalar multiplier gamma.
    tol : float, optional
        Tolerance for deciding if s_j is "zero".

    Returns
    -------
    f_val : float
        The objective value f(s) = 0.5 * Y^T M^{-1} Y.
    grad : np.ndarray, shape (p,)
        The gradient of f(s).  grad[j] = - (gamma/2) * (x_j^T M^-1 Y)^2.
    """
    n, p = X.shape
    
    # Identify which indices have nonzero s_j
    support = np.flatnonzero(np.abs(s) > tol)
    k = len(support)
    
    # If s is all zeros, then M = I => M^-1 = I => easy to compute
    if k == 0:
        f_val = 0.5 * Y.dot(Y)
        # grad_j = - (gamma/2) * (x_j^T Y)^2
        Xu = X.transpose().dot(Y)  # shape (p,)
        grad = -0.5 * gamma * (Xu**2)
        return f_val, grad

    # ---------------------------------------------------------------------
    # Build U = [ sqrt(gamma*s_j) * x_j ]_j in the support
    #
    # M = I + sum_{j in support} gamma * s_j * x_j x_j^T
    #   = I + U U^T, where U is n x k (dense).
    # ---------------------------------------------------------------------
    U = np.zeros((n, k), dtype=np.float64)
    for i, j in enumerate(support):
        coef = np.sqrt(gamma * s[j])
        # Extract column j from X in dense format
        # shape (n,1) => ravel() => (n,)
        col_j = X.getcol(j).toarray().ravel()
        U[:, i] = coef * col_j

    # ---------------------------------------------------------------------
    # Woodbury identity:
    #   (I + U U^T)^{-1} = I - U (I + U^T U)^{-1} U^T
    # => M^-1 Y = Y - U (I + U^T U)^{-1} (U^T Y)
    # ---------------------------------------------------------------------
    alpha = U.T.dot(Y)                   # shape (k,)
    S = np.eye(k) + U.T.dot(U)           # shape (k, k)
    S_inv = np.linalg.inv(S)             # shape (k, k)
    tmp = S_inv.dot(alpha)               # shape (k,)
    My = Y - U.dot(tmp)                  # shape (n,)

    # Objective = 0.5 * Y^T (M^-1 Y)
    f_val = 0.5 * Y.dot(My)

    # ---------------------------------------------------------------------
    # Gradient:
    #   grad[j] = - (gamma/2) * ( x_j^T (M^-1 Y) )^2
    #
    # We'll compute all x_j^T (M^-1 Y) at once by X^T My.
    # ---------------------------------------------------------------------
    Xu = X.transpose().dot(My)  # shape (p,)
    grad = -0.5 * gamma * (Xu**2)

    return f_val, grad

def linear_relaxation_round(M_all,c_all,blocks,y,nonzero, lambda_ridge,a_array = [], tol = 10**-2):
    regressor = gp.Model()
    regressor.params.OutputFlag = 0
    regressor.params.timelimit = 600
    regressor.params.mipgap = 0.001    


    ntrees = len(M_all)
    dim = np.sum([M1.shape[1] for M1 in M_all])
    c_ensemble = [c for c_array in c_all for c in c_array]
    M_ensemble = scipy.sparse.hstack(M_all, format='csc')

    nu = regressor.addVar(name="nu") 
    z_new = regressor.addMVar(dim,  vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="z_new") 
    z_new1 = z_new.tolist()
    regressor.setObjective(nu, GRB.MINIMIZE)
    regressor.addConstr(nu>=0)
    for t in range(ntrees):
        z_inds = np.array(blocks[t])
        for i in z_inds:
            regressor.addConstr(gp.quicksum([z_new1[j] for j in z_inds[c_ensemble[i]]]) <= (1 - z_new1[i])*len(c_ensemble[i]))
    
    if len(a_array) == 0:
        regressor.addConstr(gp.quicksum(z_new) <= nonzero)
    else:
        a_all = np.array([elem for sublist in a_array for elem in sublist])
        regressor.addConstr(z_new@a_all <=nonzero)
        
    t1 = time.time()
    cost = 10**9
    nu1 = 0
    while  (nu1 - cost)/cost < -tol:
        regressor.optimize()
        z = np.array([z_new[i].X for i in range(dim)])
        nu1 = nu.X
        cost, subgradient = objective_and_gradient_sparse_csc(M_ensemble,y,z,lambda_ridge)  
        regressor.addConstr(nu >= cost +subgradient@(z_new - z) )
        print(nu1,cost)

    z_array = []
    z_final = []

    for t in range(ntrees):
        z_tree = z[blocks[t]]
        if sum(z_tree) > 0:
            z_rounded = get_integer_soln(z_tree, c_all[t]).astype(int)
            z_array.append(z_rounded)
            z_final = np.append(z_final,z_rounded)
        else:
            z_array.append(z_tree)
            z_final = np.append(z_final,z_tree) 

    z_final = np.array(z_final).astype(int)
    w_final = np.zeros(len(z_final))


    M_sub = M_ensemble[:, z_final == 1]
    reg_value = 1.0 / lambda_ridge
    I_reg = sp.diags([reg_value] * M_sub.shape[1])
    XtX = M_sub.T @ M_sub 
    rhs = M_sub.T @ y  
    w_sub = spsolve(XtX + I_reg, rhs)
    w_final[z_final==1] = w_sub

    return z_final, w_final, z_array
