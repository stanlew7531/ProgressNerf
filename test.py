import torch
import ProgressNerf.Raycasting.csrc as _C

sigmas = torch.ones([1,2,10]).cuda()
sigmas[0,:,5:] += 2
sigmas = sigmas.repeat((3,1,1)) # (3, 2, 10)
distances = torch.vstack((torch.linspace(3,4,10), torch.linspace(1,7,10))).unsqueeze(0).repeat((3,1,1)).cuda() # (3, 2, 10)

ray_origins = torch.tensor([[0.,0.,0.],[1.,1.,1.]]).unsqueeze(0).cuda().repeat((3,1,1))
ray_directions = torch.tensor([[0., 0., 1. ], [2 ** 0.5, 2 ** 0.5, 2 ** 0.5]]).unsqueeze(0).cuda().repeat((3,1,1))

print(ray_origins.shape)
print(ray_directions.shape)

sigmas = torch.load("sigmas.pt").cuda()
distances = torch.load("distances.pt").cuda()
ray_origins = torch.load("ray_origins.pt").cuda()
ray_directions = torch.load("ray_dirs.pt").cuda()

asdf = _C.DoWeightedResampling(sigmas, distances, ray_origins, ray_directions, 5)

print(asdf)