import torch
from flcore.clients.clientbase import Client 

class FedGFSAM:
    """
    SAM helper for FedGF (interpolated perturbation).
    Implements the 2-pass routine:
      1) closure() + backward() at w -> g
      2) Build:
           w_tilde_i = w_t + rho * g/||g||               (local pert)
           w_tilde_g = w_t + rho * Δ^r/||Δ^r||           (global pert, from server)
           w_bar     = c * w_tilde_g + (1-c) * w_tilde_i (interpolation)
         Temporarily perturb weights to w_bar, run closure() + backward(), then restore.

    After this call, grads on the model are ∇F_i(w_bar).
    Caller (client) should then optionally clip and call optimizer.step().

    Args:
      model : nn.Module
      rho   : SAM radius (float), default 0.05
      eps   : small constant to avoid division-by-zero
    """    
    def __init__(self, model, base_optimizer, rho: float = 0.05, c_val: float = 0.5,  eps: float = 1e-12):
        self.model = model
        self.opt = base_optimizer
        self.rho = float(rho)
        self.eps = float(eps)
        self.c_val = float(c_val)

    @torch.no_grad()
    def params_to_vector(self, model):
        params = list(model.parameters())
        if not params:
            return torch.empty(0)
        return torch.cat([p.detach().view(-1).cpu() for p in params])

    # Concatenate grads of all trainable params into a single 1-D CPU tensor.
    def _flat_grads_cpu(self) -> torch.Tensor:
        chunks = []
        for p in self.model.parameters():
            if p.grad is None:
                chunks.append(torch.zeros(p.numel(), dtype=torch.float32, device="cpu"))
            else:
                chunks.append(p.grad.detach().to("cpu", dtype=torch.float32).view(-1))
        return torch.cat(chunks, dim=0) if chunks else torch.zeros(0, dtype=torch.float32, device="cpu")
 
    # In-place: w <- w + vec (vec is flattened on the same device as parameters)
    @torch.no_grad()
    def _add_flat_vector_(self, vec_flat_device: torch.Tensor) -> None:
        offset = 0
        for p in self.model.parameters():
            n = p.numel()
            if n == 0:
                continue
            p.add_(vec_flat_device[offset:offset + n].view_as(p))
            offset += n

    def step(self, closure, global_perturb):

        # Current model w_i_k^r snapshot (CPU)
        curr_flat = self.params_to_vector(self.model).detach().cpu()

        # ---- Compute gradients at current w ----
        loss_plain = closure()
        loss_plain.backward()
        g_flat = self._flat_grads_cpu()  # CPU flat gradient 

        # ---- build local perturbed point: w_tilde_i = w_t + rho * g/||g|| ----
        g_norm = g_flat.norm(p=2)
        if g_norm.item() > 0:
            w_tilde_i_flat = curr_flat + (self.rho / (g_norm + self.eps)) * g_flat
        else:
            w_tilde_i_flat = curr_flat.clone()

        # ---- interpolate: w_bar = c * w_tilde_g + (1-c) * w_tilde_i ----
        w_bar_flat = self.c_val * global_perturb + (1.0 - self.c_val) * w_tilde_i_flat

        # ---- Gradient at perturbed weights: pertubed g ----
        device = next(self.model.parameters()).device
        delta_flat_cpu = w_bar_flat - curr_flat
        delta_flat = delta_flat_cpu.to(device=device, dtype=torch.float32)

        # w <- w + Δ
        with torch.no_grad():
            self._add_flat_vector_(delta_flat)

        loss_adv = closure()
        loss_adv.backward()

        # restore: w <- w - Δ
        with torch.no_grad():
            self._add_flat_vector_(-delta_flat)

        return loss_plain.item(), loss_adv.item()


class clientFedGF(Client):
    def __init__(self, args, id, train_x, train_y, **kwargs):
        super().__init__(args, id, train_x, train_y, **kwargs)
        self.global_perturb = None
        self.rho = args.fedsam_rho
        self.c_val = args.c_value

    def train(self, global_round):
        self.model.train()
        self.model = self.model.to(self.device)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.local_learning_rate * (self.lr_decay ** global_round),
                                    weight_decay=self.weight_decay,momentum=self.momentum)
        fedgfsam = FedGFSAM(self.model, optimizer, rho=self.rho)
        self.trainloader = self.load_train_data()

        # --- get initial state (w_i_0^r) ---
        self.theta0 = self.params_to_vector(self.model).detach()

        for k in range(self.local_epochs):
            epoch_loss, correct, total = 0.0, 0, 0
            for xb, yb in self.trainloader:
                xb = xb.to(self.device, non_blocking=True).float()
                yb = yb.to(self.device, non_blocking=True).long().view(-1)

                def closure():
                    optimizer.zero_grad(set_to_none=True)
                    outputs = self.model(xb)
                    return self.loss(outputs, yb)

                loss1, loss2 = fedgfsam.step(closure, self.global_perturb)

                # Clip gradients to avoid explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                epoch_loss += loss2 * yb.numel()
                with torch.no_grad():
                    out_now = self.model(xb) 
                    preds = out_now.argmax(1)
                    correct += (preds == yb).sum().item()
                    total += yb.numel()
            
            avg_loss = epoch_loss / max(total, 1)
            acc = correct / max(total, 1)
            print(f"     [Client {self.id}] | Epoch {k+1}/{self.local_epochs} | Loss={avg_loss:.4f} | Acc={acc:.4f}")

        # --- Get final state (w_i_K^r) and compute delta (Δw_i^r = w^r - w_i_K^r) ---
        self.thetaK = self.params_to_vector(self.model).detach()
        self.delta_state = self.theta0 - self.thetaK



