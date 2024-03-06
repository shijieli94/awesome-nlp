import math

import numpy as np


def compute_dpp_kernel(faiss_index, embed_list, candidates_list, scale_factor):
    near_reps_list = np.array(
        [[faiss_index.index.reconstruct(int(i)) for i in candidates] for candidates in candidates_list]
    )

    # normalize first
    embed_list = embed_list / np.linalg.norm(embed_list, keepdims=True, axis=-1)
    near_reps_list = near_reps_list / np.linalg.norm(near_reps_list, keepdims=True, axis=-1)

    rel_scores_list = np.einsum("bd,bkd->bk", embed_list, near_reps_list)

    rel_scores_list = (rel_scores_list + 1) / 2  # to make kernel-matrix non-negative
    rel_scores_list -= rel_scores_list.max(keepdims=True, axis=-1)  # to prevent overflow error

    # to balance relevance and diversity
    rel_scores_list = np.exp(rel_scores_list / (2 * scale_factor))
    sim_matrix_list = np.matmul(near_reps_list, near_reps_list.transpose(0, 2, 1))

    # to make kernel-matrix non-negative
    sim_matrix_list = (sim_matrix_list + 1) / 2
    kernel_matrix_list = rel_scores_list[:, None, :] * sim_matrix_list * rel_scores_list[:, :, None]

    return near_reps_list, rel_scores_list, kernel_matrix_list


def fast_map_dpp(kernel_matrix, max_length):
    """
    fast implementation of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param max_length: positive int
    :return: list
    reference: https://github.com/laming-chen/fast-map-dpp/blob/master/dpp_test.py
    paper: Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity
    """
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(int(selected_item))
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        selected_item = np.argmax(di2s)
        selected_items.append(int(selected_item))
    return selected_items
