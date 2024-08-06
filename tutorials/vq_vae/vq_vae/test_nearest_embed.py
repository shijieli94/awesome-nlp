import unittest

import numpy as np
import torch
from torch.autograd import Variable

from .nearest_embed import nearest_embed


class NearestEmbed2dTest(unittest.TestCase):
    def test_single_embedding(self):
        # inputs
        emb = Variable(torch.eye(5, 7).float(), requires_grad=True)

        a = np.array(
            [
                [[0.9, 0.0], [0.0, 0.0], [0.0, 2.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.7], [0.0, 0.0], [0.0, 0.0], [0.6, 0.0], [0.0, 0.0]],
            ],
            dtype=np.float32,
        )

        # expected results
        out = np.array(
            [
                [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
            ],
            dtype=np.float32,
        )

        grad_input = np.array(
            [
                [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
            ],
            dtype=np.float32,
        )

        grad_emb = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        grad_input = torch.from_numpy(grad_input).float()
        grad_emb = torch.from_numpy(grad_emb).float()

        input = Variable(torch.from_numpy(a).float(), requires_grad=True)
        z_q, _ = nearest_embed(input, emb)

        (0.5 * z_q.pow(2)).sum().backward(retain_graph=True)
        out = torch.from_numpy(out)

        self.assertEqual(True, torch.equal(z_q.data, out))
        self.assertEqual(True, torch.equal(input.grad.data, grad_input))
        self.assertEqual(True, torch.equal(emb.grad.data, grad_emb))


if __name__ == "__main__":
    unittest.main()
