
import torch
import torchcsprng as csprng




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


key = torch.empty(16, dtype=torch.uint8, device=device).random_(0, 256)

print(key)
