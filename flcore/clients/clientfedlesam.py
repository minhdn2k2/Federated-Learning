import torch
from flcore.clients.clientbase import Client 

class LESAM:
    def __init__(self, model, base_optimizer, rho: float = 0.05,  eps: float = 1e-12):
        self.model = model
        self.opt = base_optimizer
        self.rho = float(rho)
        self.eps = float(eps)
        self.perturb_flat = None

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

    def step(self, closure, w_t_flat, w_old_flat):
        if w_old_flat is None:
            w_old_flat = torch.zeros_like(w_t_flat)

        # Compute the perturbation        
        device = next(self.model.parameters()).device
        d_flat = w_old_flat - w_t_flat
        d_flat_norm = d_flat.norm(p=2)

        # δ_i_k^r = p * (w_i^old-w^r) / ||w_i^old-w^r||
        if d_flat_norm.item() > 0:
            perturb_flat_cpu = (self.rho / (d_flat_norm + self.eps)) * d_flat
        else:
            perturb_flat_cpu = torch.zeros_like(d_flat)

        self.perturb_flat = perturb_flat_cpu.to(device=device, dtype=torch.float32)

        # w <- w + Δ
        with torch.no_grad():
            self._add_flat_vector_(self.perturb_flat)
        loss = closure()
        loss.backward()
        # Clip gradients to avoid explosion
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)

        # restore: w <- w - Δ
        with torch.no_grad():
            self._add_flat_vector_(-self.perturb_flat)

        # perform the base optimizer update
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)

        # cleanup
        self.perturb_flat = None

        return loss.item()


class clientFedLESAM(Client):
    def __init__(self, args, id, train_x, train_y, **kwargs):
        super().__init__(args, id, train_x, train_y, **kwargs)
        self.rho = args.fedsam_rho
        self.w_old = None

    def train(self, global_round):
        self.model.train()
        self.model = self.model.to(self.device)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.local_learning_rate * (self.lr_decay ** global_round),
                                    weight_decay=self.weight_decay,momentum=self.momentum)
        lesam = LESAM(self.model, optimizer, rho=self.rho)

        self.trainloader = self.load_train_data()

        # --- get initial state (w_i_0^r) ---
        self.theta0 = self.params_to_vector(self.model).detach()

        for k in range(self.local_epochs):
            epoch_loss, correct, total = 0.0, 0, 0
            for xb, yb in self.trainloader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True).long().view(-1)

                def closure():
                    optimizer.zero_grad(set_to_none=True)
                    outputs = self.model(xb)
                    return self.loss(outputs, yb)

                loss = lesam.step(closure, self.theta0, self.w_old)    # loss: at w+e(w)

                epoch_loss += loss * yb.numel()
                
                with torch.no_grad():
                    out_now = self.model(xb) 
                    preds = out_now.argmax(1)
                    correct += (preds == yb).sum().item()
                    total += yb.numel()

            avg_loss = epoch_loss / max(total, 1)
            acc = correct / max(total, 1)
            print(f"     [Client {self.id}] | Epoch {k+1}/{self.local_epochs} | Loss={avg_loss:.4f} | Acc={acc:.4f}")

        # --- Get final state (w_i_K^r) and compute delta (Δw_i^r = w_i_K^r − w^r) ---
        self.thetaK = self.params_to_vector(self.model).detach()
        self.delta_state = self.thetaK - self.theta0

        # Update w_i_old
        self.w_old = self.theta0

