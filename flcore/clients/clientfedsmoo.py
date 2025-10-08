import torch
from flcore.clients.clientbase import Client

class FedSMOOSAM:
    def __init__(self, model, base_optimizer, rho: float = 0.05, eps: float = 1e-12):
        self.model = model
        self.opt = base_optimizer
        self.rho = float(rho)
        self.eps = float(eps)
        self._e_ws = None  # stores per-parameter perturbations from first_step

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

    
    def step(self, closure, mu_flat_cpu: torch.Tensor, s_global_flat_cpu: torch.Tensor,) -> tuple: 
        """
        Run the FedSMOO-SAM two-pass, and compute perturbed gradient at w + ŝ_i_k^r

        Args:
          closure              : function that zeroes grads, runs forward, returns scalar loss
          mu_flat_cpu          : local dual μ_i (flattened, CPU tensor)
          s_global_flat_cpu    : global perturbation s^t from server (flattened, CPU tensor)

        Returns:
          (loss_plain, loss_adv, s_hat_flat_cpu)
        """  
        # ---- Compute gradients at current w ----
        loss_plain = closure()
        loss_plain.backward()
        g_flat = self._flat_grads_cpu()  # CPU flat gradient

        # ---- Build local perturbation ŝ_i_k^r from (g, μ_i_k^r, s^r) ----
        u = (g_flat - mu_flat_cpu) - s_global_flat_cpu
        u_norm = u.norm(p=2)
        if u_norm.item() > 0:
            s_hat_flat_cpu = (self.rho / (u_norm + self.eps)) * u
        else:
            s_hat_flat_cpu = torch.zeros_like(u)
        # -> update μ_i_{k+1}^r later

        # ---- Gradient at perturbed weights: pertubed g ----
        device = next(self.model.parameters()).device
        s_hat_dev = s_hat_flat_cpu.to(device=device, dtype=torch.float32)

        # w <- w + ŝ
        with torch.no_grad():
            self._add_flat_vector_(s_hat_dev)

        loss_adv = closure()
        loss_adv.backward()

        # restore: w <- w - ŝ
        with torch.no_grad():
            self._add_flat_vector_(-s_hat_dev)
        
        return loss_plain.item(), loss_adv.item(), s_hat_flat_cpu
    



class clientFedSMOO(Client):
    def __init__(self, args, id, train_x, train_y, **kwargs):
        super().__init__(args, id, train_x, train_y, **kwargs)
        self.rho = args.fedsam_rho
        self.beta = args.beta_fedsmoo

        self.lambda_i = None  # local dual for weights (CPU flat)
        self.mu_i     = None  # local dual for perturbation (CPU flat)
        self.global_s = None

    def get_global_s(self, global_s_flat):
        self.global_s = torch.as_tensor(global_s_flat, dtype=torch.float32, device="cpu").clone()

    # grad <- grad - lambda_i + (1/beta)(w - w^t)
    def _add_dynamic_regularizer_(self, theta0_cpu):
        offset = 0
        for p in self.model.parameters():
            n = p.numel()
            if p.grad is not None:
                lam = self.lambda_i[offset:offset+n].view_as(p).to(p.grad.device, dtype=p.grad.dtype)
                w0 = theta0_cpu[offset:offset+n].view_as(p).to(p.grad.device, dtype=p.grad.dtype)
                prox = (1.0 / self.beta) * (p.data - w0)
                p.grad.add_(-lam).add_(prox)
            offset += n
            

    def train(self, global_round):
        self.model.train()
        self.model = self.model.to(self.device)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.local_learning_rate * (float(0.9995) ** global_round),
                                    weight_decay=self.weight_decay,momentum=self.momentum)
        dysam = FedSMOOSAM(self.model, optimizer, rho=self.rho)

        self.trainloader = self.load_train_data()

        # --- get initial state (w_i_0^r) ---
        self.theta0 = self.params_to_vector(self.model).detach()

        # init duals if needed
        if self.lambda_i is None: self.lambda_i = torch.zeros_like(self.theta0)
        if self.mu_i     is None: self.mu_i     = torch.zeros_like(self.theta0)

        # s^t from server or zeros
        global_s = torch.zeros_like(self.theta0) if self.global_s is None else self.global_s.clone()


        for k in range(self.local_epochs):
            epoch_loss, correct, total = 0.0, 0, 0
            for xb, yb in self.trainloader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True).long().view(-1)

                def closure():
                    optimizer.zero_grad(set_to_none=True)
                    outputs = self.model(xb)
                    return self.loss(outputs, yb)
                
                loss1, loss2, s_hat = dysam.step(
                        closure=closure,                 # zero_grad + forward + return loss (mean)
                        mu_flat_cpu=self.mu_i,           # local μ_i (CPU flat)
                        s_global_flat_cpu=global_s)      # server s^t (CPU flat; zeros if None)

                # Update μ_i (dual for perturbation): μ_i_{k+1}^r = μ_i_k^r + (ŝ_i_k^r - s^r)
                self.mu_i.add_(s_hat - global_s)
                last_s_hat = s_hat.clone()

                # Add dynamic regularizer to current grads: grad <- grad - lambda_i + (1/beta)(w_i_k^r - w^r)
                self._add_dynamic_regularizer_(theta0_cpu=self.theta0)

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
        
        # --- Get final state (w_i_K^r) and compute delta (Δw_i^r = w_i_K^r − w^r) ---
        self.thetaK = self.params_to_vector(self.model).detach()
        self.delta_state = self.thetaK - self.theta0

        # local dual update for weights
        self.lambda_i.add_( - (1.0 / self.beta) * self.delta_state)

        # Send back s_e to server
        self.s_e = (self.mu_i - last_s_hat).to(torch.float32)