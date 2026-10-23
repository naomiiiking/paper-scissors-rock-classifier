## Paper, scissors, rock classifier

A TorchVision model trained on images of hands being paper, scissors or rock. Part of my PyTorch Fundamentals (LFD2000) course.

To run:

```bash
git clone 
git clone

uv sync
uv run pipeline.py
```

Example output:
```bash
Test accuracy: 58.06%
Confusion matrix:
tensor([[ 6, 23, 50],
        [ 6, 27, 48],
        [12, 22, 46]])
     paper: precision = 0.25, recall = 0.08
      rock: precision = 0.38, recall = 0.33
  scissors: precision = 0.32, recall = 0.57
```