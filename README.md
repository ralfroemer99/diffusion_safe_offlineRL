This repository contains the code for the paper "Safe Offline Reinforcement Learning using Trajectory-Level Diffusion Models", which was presented at the ICRA2024 Workshop ["Back to the Future: Robot Learning Going Probabilistic"](https://probabilisticrobotics.github.io/). The paper can be found [here](https://openreview.net/pdf?id=o575pIMeE).

## Install on Ubuntu

```bash
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

## Run

```bash
python scripts/generate_dataset.py
python scripts/train.dy
python scripts/eval.py
```

## Reference

```bash
@inproceedings{romer2024safe,
  title={Safe Offline Reinforcement Learning using Trajectory-Level Diffusion Models},
  author={R{\"o}mer, Ralf and Brunke, Lukas and Schuck, Martin and Schoellig, Angela P},
  booktitle={ICRA 2024 Workshop Back to the Future: Robot Learning Going Probabilistic}
}
```

## Acknowledgements
We use some code from [Diffuser](https://github.com/jannerm/diffuser). 
