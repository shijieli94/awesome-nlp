import logging

import numpy as np

from ..utils.dpp import compute_dpp_kernel, fast_map_dpp

logger = logging.getLogger(__name__)


class Searcher:
    def __init__(self, mode, topk, scale_factor=0.05, full_random=False):
        self.mode = mode
        self.topk = topk
        self.scale_factor = scale_factor
        self.full_random = full_random  # random choose from full space instead of KNN-reduced space

    def __call__(self, *args, **kwargs):
        return getattr(self, self.mode)(*args, **kwargs)

    def knn(self, res_list, faiss_index, num_ice, num_candidates=1):
        embed_list = np.stack([res["embed"] for res in res_list])
        nearest_ids = faiss_index.search(embed_list, num_ice)[1].tolist()

        for i in range(len(res_list)):
            res_list[i] = res_list[i]["entry"]
            res_list[i]["ctxs"] = nearest_ids[i]
            res_list[i]["ctxs_candidates"] = [nearest_ids[i]]

        return res_list

    def random(self, res_list, faiss_index, num_ice, num_candidates=1):
        # first reduce search space by knn
        candidates_list = np.stack([res["ctxs"] for res in self.knn(res_list, faiss_index, num_ice=self.topk)])

        for i in range(len(res_list)):
            ctxs_candidates = [candidates_list[i][:num_ice].tolist()]  # always include the knn results
            while len(ctxs_candidates) < num_candidates:
                samples_ids = np.random.choice(
                    faiss_index.ntotal if self.full_random else candidates_list[i], num_ice, replace=False
                )
                samples_ids = sorted(samples_ids.tolist())
                if samples_ids not in ctxs_candidates:
                    ctxs_candidates.append(samples_ids)

            assert len(ctxs_candidates) == num_candidates

            res_list[i]["ctxs"] = ctxs_candidates[0]
            res_list[i]["ctxs_candidates"] = ctxs_candidates

        return res_list

    def dpp(self, res_list, faiss_index, num_ice, num_candidates=1):
        embed_list = np.stack([res["embed"] for res in res_list])
        # first reduce search space by knn
        candidates_list = np.stack([res["ctxs"] for res in self.knn(res_list, faiss_index, num_ice=self.topk)])

        near_reps_list, rel_scores_list, kernel_matrix_list = compute_dpp_kernel(
            faiss_index, embed_list, candidates_list, self.scale_factor
        )

        if np.isinf(kernel_matrix_list).any() or np.isnan(kernel_matrix_list).any():
            logger.info("Inf or NaN detected in kernel matrix, using knn results instead!")
            ctxs_candidates_idx_list = [[list(range(num_ice))] for _ in kernel_matrix_list]
        else:  # MAP inference
            ctxs_candidates_idx_list = [
                [sorted(fast_map_dpp(kernel_matrix, num_ice))] for kernel_matrix in kernel_matrix_list
            ]

        for i in range(len(res_list)):
            candidates = candidates_list[i]
            ctxs_candidates_idx = ctxs_candidates_idx_list[i]
            ctxs_candidates = [[candidates[j] for j in ctxs_idx] for ctxs_idx in ctxs_candidates_idx]
            assert len(ctxs_candidates) == num_candidates

            res_list[i]["ctxs"] = ctxs_candidates[0]
            res_list[i]["ctxs_candidates"] = ctxs_candidates

        return res_list

    def kdpp(self, res_list, faiss_index, num_ice, num_candidates=1):
        try:
            from dppy.finite_dpps import FiniteDPP
        except ImportError:
            raise ImportError("kdpp requires dppy package.")

        embed_list = np.stack([res["embed"] for res in res_list])
        # first reduce search space by knn
        candidates_list = np.stack([res["ctxs"] for res in self.knn(res_list, faiss_index, num_ice=self.topk)])

        near_reps_list, rel_scores_list, kernel_matrix_list = compute_dpp_kernel(
            faiss_index, embed_list, candidates_list, self.scale_factor
        )

        for i, (kernel_matrix, rel_scores) in enumerate(zip(kernel_matrix_list, rel_scores_list)):
            ctxs_candidates_idx = [list(range(num_ice))]
            dpp_L = FiniteDPP("likelihood", **{"L": kernel_matrix})
            while len(ctxs_candidates_idx) < num_candidates:
                try:
                    samples_ids = np.array(
                        dpp_L.sample_exact_k_dpp(size=num_ice, random_state=len(ctxs_candidates_idx))
                    )
                except Exception as e:
                    logger.warning(e)
                    continue
                # ordered by relevance score
                samples_scores = np.array([rel_scores[k] for k in samples_ids])
                samples_ids = samples_ids[(-samples_scores).argsort()].tolist()

                if samples_ids not in ctxs_candidates_idx:
                    assert len(samples_ids) == num_ice
                    ctxs_candidates_idx.append(samples_ids)

            ctxs_candidates = [[candidates_list[i][k] for k in ctxs_idx] for ctxs_idx in ctxs_candidates_idx]
            assert len(ctxs_candidates) == num_candidates

            res_list[i]["ctxs"] = ctxs_candidates[0]
            res_list[i]["ctxs_candidates"] = ctxs_candidates

        return res_list
