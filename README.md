## Requirements
- PyTorch (1.0+)
- python 3

## Installation 
```shell
pip install pytorch-seed
```
You can also install in editable mode with `python3 -m pip install -e .` so that modifications
in the repository are automatically synced with the installed library.

## Usage
Similar to [pytorch lightning's seed_everything](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.utilities.seed.html),
we have
```python
import pytorch_seed
pytorch_seed.seed(123)
```
which will seed python's base RNG, numpy's RNG, torch's CPU RNG, and all CUDA RNGs.

Also similar to pytorch lightning's `isolate_rng` context manager, we have
```python
import torch
import pytorch_seed

pytorch_seed.seed(1)
with pytorch_seed.SavedRNG():
    print(torch.rand(1)) # tensor([0.7576])
print(torch.rand(1)) # tensor([0.7576])
```

They can also be used to maintain independent RNG streams:
```python
import torch
import pytorch_seed

rng_1 = pytorch_seed.SavedRNG(1) # start the RNG stream with seed 1
rng_2 = pytorch_seed.SavedRNG(2)

with rng_1:
    # does not affect, nor is affected by the global RNG and rng_2
    print(torch.rand(1)) # tensor([0.7576])

with rng_2:
    print(torch.rand(1)) # tensor([0.6147])

torch.rand(1) # modify the global RNG state

with rng_1:
    # resumes from the last context
    print(torch.rand(1)) # tensor([0.2793])

with rng_2:
    print(torch.rand(1)) # tensor([0.3810])
    
# confirm those streams are the uninterrupted ones
pytorch_seed.seed(1)
torch.rand(2) # tensor([0.7576, 0.2793])

pytorch_seed.seed(2)
torch.rand(2) # tensor([0.6147, 0.3810])
```
## Testing
Install `pytest` if you don't have it, then run
```
py.test
```
