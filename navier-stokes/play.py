import torch
from utils import compute_avg_pixel_norm

def main():
    path = '/scratch/qingqu_root/qingqu/jiayx/FlowDAS/data_file.pt'
    data, time = torch.load(path)


    # for each image, compute the norm and then average over all images
    avg_pixel_norm = torch.mean(torch.sqrt(torch.sum(data ** 2, dim=(1, 2, 3))))


    print(avg_pixel_norm.item())

if __name__ == "__main__":
    main()