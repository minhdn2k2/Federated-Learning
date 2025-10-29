# Federated Learning in PyTorch

A lightweight, self‑contained Federated Learning (FL) toolbox built **only on PyTorch**.  
This repository trains *multiple algorithms sequentially* on a chosen dataset and saves learning curves for easy comparison.

---

## Implemented Algorithms
- **FedAvg** — parameter averaging
- **SCAFFOLD** — control variates to counter client drift
- **FedDyn** — dynamic regularization on server
- **FedSAM** — SAM inside FL: sharpness‑aware local updates
- **MoFedSAM** — momentum/EMA flavored SAM in FL
- **FedSMOO** — smoothness/variance‑aware variant
- **FedGF** — gradient filtering / geometry‑guided perturbation
- **FedLESAM** — low/enhanced sharpness variants

> Exact behaviors reflect this repo’s implementation; naming may differ slightly from original papers.

---

## Repository Structure
```
Federated-Learning/
├─ main.py                      # Entry: trains all algorithms sequentially, plots & saves results
├─ flcore/
│  ├─ servers/                  # Server implementations per algorithm
│  │   ├─ serverbase.py
│  │   ├─ serveravg.py
│  │   ├─ serverscaffold.py
│  │   ├─ serverfeddyn.py
│  │   ├─ serverfedsam.py
│  │   ├─ servermofedsam.py
│  │   ├─ serverfedsmoo.py
│  │   ├─ serverfedgf.py
│  │   └─ serverfedlesam.py
│  └─ clients/                  # Client implementations per algorithm
│      ├─ clientbase.py
│      ├─ clientavg.py
│      ├─ clientscaffold.py
│      ├─ clientfeddyn.py
│      ├─ clientfedsam.py
│      ├─ clientmofedsam.py
│      ├─ clientfedsmoo.py
│      ├─ clientfedgf.py
│      └─ clientfedlesam.py
├─ utils/
│  ├─ utils_dataset.py          # Dataset loading, partitioning (IID/Dirichlet), caching under Data/
│  ├─ utils_model.py            # CNN (default) and ResNet18+GroupNorm (available)
│  └─ loss_landscape.py         # (optional utilities)
├─ Output/                      # Plots (.png) & history dumps (.npy) will be saved here
└─ Data/                        # Auto‑downloaded raw data & preprocessed splits are cached here
```

---

## Quick Start
Train **all algorithms** on **CIFAR‑10** with Dirichlet non‑IID (α=0.3), 100 clients, 10% participation, for 800 rounds:
```bash
python main.py 
```
If you want to change the hyperparameters:
```bash
python main.py \
  --dataset_name CIFAR10 \
  --n_clients 100 \
  --rule Dirichlet --rule_arg 0.3 \
  --selected_ratio 0.1 \
  --global_rounds 800 \
  --local_epochs 5 \
  --batch_size 50 \
  --local_learning_rate 0.1 \
  --weight_decay 1e-4 \
  --device cuda \
  --seed 42
```

> This will sequentially run **FedAvg, SCAFFOLD, FedDyn, FedSAM, FedSMOO, FedLESAM, FedGF, MoFedSAM** (where applicable) and save results under `Output/`.

**Change dataset**:
- `--dataset_name mnist`
- `--dataset_name CIFAR10`
- `--dataset_name CIFAR100`

> Datasets are automatically downloaded into `Data/Raw` on first run.

---

## Command‑line Arguments
Below are the most important flags (default values in **bold**):

### Dataset & Partitioning
- `--dataset_name {mnist|CIFAR10|CIFAR100}` (default: **CIFAR10**)
- `--n_clients INT` total clients (default: **100**)
- `--rule {Dirichlet}` partition rule (default: **Dirichlet**)
- `--rule_arg FLOAT` Dirichlet α (default: **0.1**)
- `--unbalanced_sgm FLOAT` optional size skew (default: **0.0**)
- `--seed INT` randomness seed (default: **42**)

### Federated Optimization
- `--global_rounds INT` communication rounds (default: **800**)
- `--selected_ratio FLOAT` fraction of clients per round (default: **0.1**)
- `--global_learning_rate FLOAT` server LR for model update (default: **1.0**)
- `--device {cpu|cuda}` training device (default: **cuda**)

### Local Training
- `--local_epochs INT` local epochs per selected client (default: **5**)  
- `--batch_size INT` local minibatch size (default: **50**)
- `--local_learning_rate FLOAT` client optimizer LR (default: **0.1**)
- `--lr_decay FLOAT` multiplicative LR decay per local epoch (default: **0.998**)
- `--weight_decay FLOAT` L2 (default: **1e-4**)
- `--momentum FLOAT` (default: **0.0**)

### Algorithm‑specific
- **FedDyn**: `--feddyn_beta FLOAT` (default: **100.0**)
- **SAM‑family (FedSAM, MoFedSAM, FedGF helper)**: `--fedsam_rho FLOAT` (default: **0.1**)
- **MoFedSAM**: `--beta_mofedsam FLOAT` (default: **0.9**)
- **FedSMOO**: `--beta_fedsmoo FLOAT` (default: **10.0**)
- **FedGF**: `--c_value FLOAT` interpolation between global/local perturbations (default: **0.5**)

> **Model**: By default `main.py` uses `utils_model.CNN` for CIFAR‑style inputs. A ResNet‑18 with GroupNorm (`Resnet18_GN`) is available in `utils_model.py` if you wish to modify `main.py` accordingly.

---

## Extending the Codebase
1. **Add a client**: create `flcore/clients/client<YourAlgo>.py` inheriting from `Client` and implement `train(self, global_round)`.
2. **Add a server**: create `flcore/servers/server<YourAlgo>.py` inheriting from `BaseServer`. Implement:
   - `setup_clients(...)` with your client class
   - `train(...)` including: select clients → send model → receive deltas → aggregate → update global model → test
3. **Wire it in**: import your server in `main.py` and insert it into the sequential training block and plotting section.
4. **(Optional) Helpers**: place reusable utilities under `utils/`.

The core server utilities you may reuse:
- `BaseServer.params_to_vector(...)`, `vector_to_params(...)`
- `setup_clients(...)`, `select_clients(...)`
- `receive_models(...)` → collects per‑client deltas & sample counts
- `get_acc_loss(...)` → quick evaluation on test data

---

## Citations

- **FedAvg** — [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629) *AISTATS 2017*
- **SCAFFOLD** — [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](https://arxiv.org/abs/1910.06378) *ICML 2020*  
- **FedDyn** — [Federated Learning Based on Dynamic Regularization](https://arxiv.org/abs/2111.04263) *ICLR 2021*
- **FedSAM, MoFedSAM** — [Generalized Federated Learning via Sharpness Aware Minimization](https://arxiv.org/abs/2206.02618) *ICML 2022*
- **FedSMOO** — [Dynamic Regularized Sharpness Aware Minimization in Federated Learning: Approaching Global Consistency and Smooth Landscape](https://arxiv.org/abs/2305.11584) *ICML 2023*
- **FedGF** — [Rethinking the Flat Minima Searching in Federated Learning](https://proceedings.mlr.press/v235/lee24aa.html) *ICML 2024*
- **FedGF** — [Locally Estimated Global Perturbations are Better than Local Perturbations for Federated Sharpness-aware Minimization](https://arxiv.org/abs/2405.18890) *ICML 2024*

---

## Acknowledgements
This codebase is built and organized by the repository authors. It relies on PyTorch/torchvision for model training and dataset handling.


