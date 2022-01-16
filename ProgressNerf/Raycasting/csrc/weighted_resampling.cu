#include "weighted_resampling.cuh"
#include <vector>

/*
Implements the weighted resampling part of the Mildenhall paper

sigmas: (batch_size, num_rays, num_samples)
distances: (batch_size, num_rays, num_samples)
ray_origins: (batch_size, num_rays, 3)
ray_directions: (batch_size, num_rays, 3)
num_samples: int

sigmas - contains the raw sigma outputs from the MLP
distances - the original distances for the coarse sampling
ray_origins - 3 items are (X, Y, Z)
ray_directions - 3 items are (delta_X, delta_Y, delta_Z) and should norm to 1
num_samples - the number of new samples to take for the fine network

returns: Tensor of shape (batch_size, num_rays, num_samples, 3)
containing the new sampled points become ray_origin + ray_direction * distance
*/
std::vector<Tensor> DoWeightedResampling(Tensor sigmas, Tensor distances, Tensor ray_origins, Tensor ray_directions, int num_resamples)
{
    Tensor weights = sigmas + 1e-5; // avoid div by zero
    Tensor pdf = weights / weights.sum(-1, true);
    // turn pdf into cdf and make sure it starts with 0.0 & ends with 1.0
    Tensor cdf = pdf.cumsum(-1);
    using namespace torch::indexing;
    cdf = torch::cat({torch::zeros_like(cdf.index({"...", Slice(None, 1)})), cdf, torch::ones_like(cdf.index({"...", Slice(None, 1)})) + 1.0e-6}, -1);
    
    // create random U values to use in the CDF inversion
    Tensor us = (torch::rand({sigmas.size(0), sigmas.size(1), num_resamples}).to(sigmas.device()) + 1e-5) * (1.0 / (1.0 + 1e-5)); // quick remap to prevent u == 0

    Tensor resamples = ray_origins.clone().unsqueeze(2).repeat({1, 1, num_resamples, 1}); //(batch_size, num_rays, num_resamples, 3)

    auto batch_size = distances.size(0);
    auto num_rays = distances.size(1);
    auto options = torch::TensorOptions().device(distances.device());
    Tensor resample_distances = torch::zeros({batch_size, num_rays, num_resamples, 3}, options); //(batch_size, num_rays, num_resamples, 3)

    // perform batched CDF inversion
    auto batch_idx = sigmas.size(0) - 1;
    for(;batch_idx >=0; batch_idx--)
    {
        Tensor us_batch = us.index({batch_idx}); // (num_rays, num_resamples)
        Tensor cdf_batch = cdf.index({batch_idx}); // (num_rays, num_samples)
        Tensor distances_batch = distances.index({batch_idx}); // (num_rays, num_samples)
        // for distances, we make sure it is prepended with a 0.0 approximator, and post-pended with a maximal distance val
        distances_batch = torch::cat({
            torch::zeros_like(distances_batch.index({"...", Slice(None, 1)})) + 1e-6,
            distances_batch,
            torch::zeros_like(distances_batch.index({"...", Slice(None, 1)})) + 1e-6 + torch::max(distances_batch)}, -1);

        // find out what bin the U goes into
        //printf("doing search sorted!!\n");
        Tensor indices = torch::searchsorted(cdf_batch, us_batch, true).to(torch::kInt64); // (num_rays, num_resamples)
        //std::cout << indices << std::endl;
        // then perform linear interpolation within the bin
        //printf("doing mincdfs!!\n");
        Tensor min_cdfs;
        min_cdfs = torch::gather(cdf_batch, 1, indices - 1); // min of bin range - (num_rays, num_resamples)
        //printf("doing maxcdfs!!\n");
        Tensor max_cdfs = torch::gather(cdf_batch, 1, indices); // max of bin range - (num_rays, num_resamples)
        //printf("doing mindistances!!\n");
        Tensor min_distances = torch::gather(distances_batch, 1, indices - 1); // min distance of bin range - (num_rays, num_resamples)
        //printf("doing maxdistances!!\n");
        Tensor max_distances = torch::gather(distances_batch, 1, indices - 0); // max distance of bin range - (num_rays, num_resamples)
        // distance is the affine transform between the two (distance) bin endpoints
        Tensor alphas = (us_batch - min_cdfs) / (max_cdfs - min_cdfs);
        Tensor resampled_distances = min_distances + alphas * (max_distances - min_distances); // (num_rays, num_resamples) here, distances are actually distances
        resampled_distances = resampled_distances.unsqueeze(-1).repeat({1,1,3}); //(num_rays, num_resamples, 3)
        resampled_distances = resampled_distances * ray_directions.index({batch_idx}).unsqueeze(1).expand({-1, num_resamples, -1}); // here, it becomes the vector from the origin

        resamples.index_put_({batch_idx}, resampled_distances + ray_origins.index({batch_idx}).unsqueeze(1).expand({-1, num_resamples, -1}));
        resample_distances.index_put_({batch_idx}, resampled_distances);

        //printf("\r\n\r\n");
    }
    std::vector<Tensor> toReturn = {};
    toReturn.push_back(resamples);
    toReturn.push_back(resample_distances.index({"...",0}));

    return toReturn;
}