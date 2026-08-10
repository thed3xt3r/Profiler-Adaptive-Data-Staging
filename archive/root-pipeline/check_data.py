# Add to check_data.py
import sys
sys.path.insert(0, '/workspace')
from dataset import TellSiteDataset, collect_samples, get_train_transforms

positives, negatives = collect_samples('/data')

ds = TellSiteDataset(positives[:10], transform=get_train_transforms())
for i in range(5):
    img, mask = ds[i]
    fg = (mask == 1).sum().item()
    total = mask.numel()
    print(f"Sample {i} | mask unique: {mask.unique()} | fg: {fg}/{total} ({fg/total*100:.3f}%)")
