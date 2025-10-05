from flcore.clients.clientbase import Client
import torch

# ------------ Optimizer ------------

class SAM:
    """
    Sharpness-Aware Minimization (SAM) wrapper.
    Use with a base optimizer (e.g., SGD/AdamW). Implements the standard 2-pass routine:
      1) First pass: compute gradients at w, then perturb weights w -> w + e(w)
      2) Second pass: compute gradients at w + e(w), restore weights, then call optimizer.step()

    Args:
      model: nn.Module
      base_optimizer: torch.optim.Optimizer (e.g., torch.optim.SGD(...))
      rho: SAM radius (float), default 0.05
      eps: small constant to avoid division by zero
    """
    def __init__(self, model, base_optimizer, rho: float = 0.05, eps: float = 1e-12):
        self.model = model
        self.opt = base_optimizer
        self.rho = float(rho)
        self.eps = float(eps)
        self._e_ws = None  # stores per-parameter perturbations from first_step

    @torch.no_grad()
    def _grad_norm(self) -> torch.Tensor:
        """Compute ||g||_2 over all available gradients in the model."""
        device = next(self.model.parameters()).device
        norms = []
        for p in self.model.parameters():
            if p.grad is not None:
                norms.append(p.grad.detach().norm(p=2))
        if not norms:
            return torch.tensor(0.0, device=device)
        return torch.norm(torch.stack(norms), p=2)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = True) -> None:
        """
        Apply perturbation: w <- w + e(w) where e(w) = rho * g / ||g||.
        Assumes: backward() has been called to populate gradients at w.
        """
        grad_norm = self._grad_norm()
        scale = (self.rho / (grad_norm + self.eps))
        self._e_ws = []
        for p in self.model.parameters():
            if p.grad is None:
                self._e_ws.append(None)
                continue
            e = p.grad * scale
            p.add_(e)              # w <- w + e
            self._e_ws.append(e)   # store the perturbation for later restore
        if zero_grad:
            self.opt.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = True) -> None:
        """
        Restore weights: w <- w - e(w), then perform optimizer.step().
        Assumes: backward() has been called on the adversarial loss at w + e(w).
        """
        if self._e_ws is None:
            raise RuntimeError("second_step() called before first_step().")
        # restore original weights
        for p, e in zip(self.model.parameters(), self._e_ws):
            if (p is not None) and (e is not None):
                p.sub_(e)
        # perform the base optimizer update
        self.opt.step()
        if zero_grad:
            self.opt.zero_grad(set_to_none=True)
        # cleanup
        self._e_ws = None

    def step(self, closure) -> tuple:
        """
        Perform a full SAM step using a closure:
          - The closure must: zero gradients inside, run forward, and return a scalar loss.
          - This method will:
              1) call closure() + backward() at w
              2) call first_step() (apply perturbation)
              3) call closure() + backward() at w + e(w)
              4) call second_step() (restore weights & optimizer.step)
        Returns:
          (loss_plain, loss_adv): the losses from the first and second passes (floats)
        """
        # Compute gradients at w
        loss_plain = closure()
        loss_plain.backward()
        # Clip gradients to avoid explosion
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)

        # Create perturbation
        self.first_step(zero_grad=True)

        # Compute gradients at w + e(w)
        loss_adv = closure()
        loss_adv.backward()
        # Clip gradients to avoid explosion
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
    
        # Restore weights and update
        self.second_step(zero_grad=True)

        return loss_plain.item(), loss_adv.item()

# -----------------------------------


class clientFedSAM(Client):
    def __init__(self, args, id, train_x, train_y, **kwargs):
        super().__init__(args, id, train_x, train_y, **kwargs)
        self.rho = args.fedsam_rho

    def train(self, global_round):
        self.model.train()
        self.model = self.model.to(self.device)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.local_learning_rate * (self.lr_decay ** global_round),
                                    weight_decay=self.weight_decay,momentum=self.momentum)
        sam = SAM(self.model, optimizer, rho=self.rho)

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

                loss1, loss2 = sam.step(closure)     # loss1: at w, loss2: at w+e(w)

                epoch_loss += loss2
                
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


