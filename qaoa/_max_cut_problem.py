# Created by Anton Bozhedarov

import numpy as np
import networkx as nx
from scipy.optimize import minimize, brute

from ._qaoa_circuit import QAOACircuit, build_qaoa_circuit


def graph_to_clause(graph: nx.graph) -> list:
    """
    constructs a clause list in the form
    [(0.5, (1, 3, 5)), (0.2, (1, 2))] represents 0.5*Z1*Z3*Z5 + 0.2*Z1*Z2
    from a matrix of a graph

    Arguments:
        graph (nd.array) - represents undirected unweighted graph

    Returns:
        clause_list (list)
    """

    # num_vert = len(graph)
    clause_list = []

    # for i in range(num_vert):
    #     for j in range(i, num_vert):
    #         if graph[i,j] != 0:
    #             clause_list.append((-0.5, (i, j)))

    for edge in graph.edges:
        i, j = edge
        clause_list.append((-0.5, (i, j)))
    return clause_list


def get_bit(z, i):
    """
    gets the i'th bit of the integer z (0 labels least significant bit)
    """
    return (z >> i) & 0x1


def max_cut_obj(z, clause_list):
    """
    Returns loss for a max cut problem 
    """
    loss = 0
    for w, (start, end) in clause_list:
        loss -= w * (1 - 2 * (get_bit(z, start) ^ get_bit(z, end)))
    return loss


def solve_graph(graph, depth, x0=None, optimizer='COBYLA', max_iter=1000, spelling=False):
    """
    solve a problem defined by a graph
    """
    num_bit = len(graph.nodes) #graph.shape[0]
    N = 2**num_bit

    clause_list = graph_to_clause(graph)
    # loss_func = lambda z: max_cut_obj(z, clause_list)

    def loss_func(z):
        return max_cut_obj(z, clause_list)
    valid_mask = None

    loss_table = np.array([loss_func(z) for z in range(N)])
    cc = build_qaoa_circuit(clause_list, num_bit, depth)

    # obtain and analyse results
    qaoa_loss, log = get_qaoa_loss(cc, loss_table, spelling=spelling) # the expectation value of loss function

    if x0 is None:
        x0 = np.zeros(cc.num_param)

    if optimizer == 'COBYLA':
        best_x = minimize(qaoa_loss,
                          x0=x0,
                          method='COBYLA',
                          options={'maxiter': max_iter}).x
    else:
        raise
    ans = qaoa_result_digest(best_x, cc, loss_table)
    # show_graph(graph, ans[2])
    return ans


def get_amplitudes(graph, depth, x0=None, optimizer='COBYLA', max_iter=1000, spelling=False):
    """
    solve a problem defined by a graph
    """
    num_bit = len(graph.nodes) #graph.shape[0]
    N = 2**num_bit

    clause_list = graph_to_clause(graph)
    # loss_func = lambda z: max_cut_obj(z, clause_list)

    def loss_func(z):
        return max_cut_obj(z, clause_list)
    valid_mask = None

    loss_table = np.array([loss_func(z) for z in range(N)])
    cc = build_qaoa_circuit(clause_list, num_bit, depth)

    # obtain and analyse results
    qaoa_loss, log = get_qaoa_loss(cc, loss_table, spelling=spelling) # the expectation value of loss function

    if x0 is None:
        x0 = np.zeros(cc.num_param)

    if optimizer == 'COBYLA':
        best_x = minimize(qaoa_loss,
                          x0=x0,
                          method='COBYLA',
                          options={'maxiter': max_iter}).x
    else:
        raise
    bs, gs = best_x[:depth], best_x[depth:]
    pl = np.abs(cc.evolve(bs, gs)) ** 2
    ans = qaoa_result_digest(best_x, cc, loss_table)
    return pl, ans


def show_loss_table(graph):
    """
    Arguments:
        graph (nx.graph):
    Returns: loss_table (ndarray)
    """
    num_bit = len(graph.nodes) #graph.shape[0]
    N = 2**num_bit

    clause_list = graph_to_clause(graph)
    # loss_func = lambda z: max_cut_obj(z, clause_list)

    def loss_func(z):
        return max_cut_obj(z, clause_list)
    valid_mask = None

    loss_table = np.array([loss_func(z) for z in range(N)])

    return loss_table


def get_qaoa_loss(circuit, loss_table, var_mask=None, x0=None, spelling=False):
    """
    obtain the loss function for qaoa.

    Args:
        circuit (QAOACircuit): quantum circuit designed for QAOA.
        loss_table: a table of loss, iterating over 2**num_bit configurations.
        var_mask (1darray, dtype='bool'): mask for training parameters.

    Returns:
        func: loss function with single parameter x.
    """
    if var_mask is None:
        var_mask = np.ones(circuit.num_param, dtype='bool')
    if x0 is None:
        x0 = np.zeros(len(var_mask))
    log = {'loss': []}

    def loss(params):
        x0[var_mask] = params
        bs, gs = x0[:circuit.depth], x0[circuit.depth:]
        psi = circuit.evolve(bs, gs)
        pl = np.abs(psi) ** 2
        exp_val = (loss_table * pl).sum()
        if log is not None:
            log['loss'].append(exp_val)
        if spelling:
            print('loss = %s' % exp_val)

        return exp_val

    return loss, log


def qaoa_result_digest(x, circuit, loss_table):
    """
    returns a quality from [0-1] of the result of the QAOA algorithm as compared to the optimal solution
    """
    num_bit, depth = circuit.num_bit, circuit.depth

    # get resulting distribution
    bs, gs = x[:depth], x[depth:]
    pl = np.abs(circuit.evolve(bs, gs)) ** 2

    # calculate losses
    max_ind = np.argmax(pl)
    mean_loss = (loss_table * pl).sum()
    most_prob_loss = loss_table[max_ind]

    # get the exact solution
    exact_x = np.argmin(loss_table)
    exact_xs = np.argwhere(loss_table == np.amin(loss_table)).flatten()
    exact_loss = loss_table[exact_x]

    print(
        'Obtain: p(%d) = %.4f with loss = %.4f, mean loss = %.4f.' % (max_ind, pl[max_ind], most_prob_loss, mean_loss))
    print('Exact x = %d, loss = %.4f.' % (exact_x, exact_loss))
    print(f'All exact solutions are {exact_xs}')
    return max_ind, most_prob_loss, exact_x, exact_loss, pl[max_ind]
