/**
 * Copyright 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <torch/extension.h>

// CUDA forward declarations
torch::Tensor ngram_repeat_block_cuda_forward(
    torch::Tensor tokens,
    torch::Tensor lprobs,
    int bsz,
    int step,
    int beam_size,
    int no_repeat_ngram_size);
