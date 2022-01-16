#include <cstdint>
#include <torch/extension.h>
#include <tuple>

#include "weighted_resampling.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{
    m.def("DoWeightedResampling", &DoWeightedResampling);
}