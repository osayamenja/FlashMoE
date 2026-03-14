import torch

if __name__ == "__main__":
    S = 8
    H = 8
    x = torch.empty((S, H), device=0, dtype=torch.float32).uniform_(-1.0, 1.0)
    x.contiguous()
    x_ptr = x.data_ptr()
    print(x)