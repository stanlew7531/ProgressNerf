#ifndef WEIGHTED_RESAMPLING
#define WEIGHTED_RESAMPLING

#include <torch/extension.h>

using torch::Tensor;

std::vector<Tensor> DoWeightedResampling(Tensor sigmas, Tensor distances, Tensor ray_origins, Tensor ray_directions, int num_samples);

#endif