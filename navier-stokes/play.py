import torch
from utils import compute_avg_pixel_norm

def main():
    path = '/scratch/qingqu_root/qingqu/jiayx/FlowDAS/data_file.pt'
    data, time = torch.load(path)


    # for each image, compute the norm and then average over all images
    avg_pixel_norm = torch.mean(torch.sqrt(torch.sum(data ** 2, dim=(1, 2, 3))))


    print(avg_pixel_norm.item())




import torch
import torch.nn.functional as F

class NoiseSampler:
    def __init__(self, noise_dist='logit_normal', P_mean=-0.4, P_std=1.0, device='cpu', dtype=torch.float32):
        self.noise_dist = noise_dist
        self.P_mean = P_mean
        self.P_std = P_std
        self.device = device
        self.dtype = dtype

    def _logit_normal_dist(self, batch_size):
        z = torch.randn((batch_size, 1, 1, 1), device=self.device, dtype=self.dtype)
        x = torch.sigmoid(z * self.P_std + self.P_mean)
        return x

    def _uniform_dist(self, batch_size):
        return torch.rand((batch_size, 1, 1, 1), device=self.device, dtype=self.dtype)

    def noise_distribution(self):
        if self.noise_dist == 'logit_normal':
            return self._logit_normal_dist
        elif self.noise_dist == 'uniform':
            return self._uniform_dist
        else:
            raise ValueError(f"Unknown noise distribution: {self.noise_dist}")

    def sample_tr(self, batch_size):
        dist = self.noise_distribution()
        t = dist(batch_size)
        r = dist(batch_size)
        t_max = torch.maximum(t, r)
        r_min = torch.minimum(t, r)
        return t_max, r_min
    
def play():
    noise_sampler = NoiseSampler(noise_dist='logit_normal', P_mean=-0.4, P_std=1.0, device='cpu', dtype=torch.float32)
    t_max, r_min = noise_sampler.sample_tr(10)
    print(t_max.shape)
    print(r_min.shape)
    print(t_max)
    print(r_min)



if __name__ == "__main__":
    play()