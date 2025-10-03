import torch
from flcore.clients.clientbase import Client


class clientScaffold(Client):
    def __init__(self, args, id, train_x, train_y, **kwargs):
        super().__init__(args, id, train_x, train_y, **kwargs)
        self.ci = None  
        self.c_global = None
        
    @torch.no_grad()
    def receive_controls(self, c_global_flat):
        """Server sends current global control variate c (flat CPU tensor)."""
        self.c_global = torch.as_tensor(c_global_flat, dtype=torch.float32, device="cpu").clone()
        if self.ci is None:
            # Initialize c_i to zeros of same shape
            self.ci = torch.zeros_like(self.c_global, device="cpu")

    def train(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.local_learning_rate,
                                    weight_decay=self.weight_decay,momentum=self.momentum)
        self.trainloader = self.load_train_data()
        
        self.model.train()
        self.model = self.model.to(self.device)

        # --- get initial state (θ^{0}) ---
        self.theta0 = self.params_to_vector(self.model)

        # Compute correction (c - c_i)
        self.c_diff = (self.c_global - self.ci).to(self.device)

        local_steps = 0
        for k in range(self.local_epochs):
            epoch_loss, correct, total = 0.0, 0, 0
            for xb, yb in self.trainloader:
                xb = xb.to(self.device, non_blocking=True).float()
                yb = yb.to(self.device, non_blocking=True).long().view(-1)

                outputs = self.model(xb)
                loss = self.loss(outputs, yb)
                optimizer.zero_grad()
                loss.backward()

                # ---- SCAFFOLD gradient correction: g <- g + (c - c_i) ----
                offset = 0
                for p in self.model.parameters():
                    n = p.numel()
                    if p.grad is not None:
                        corr = self.c_diff[offset:offset+n].view_as(p)
                        p.grad.add_(corr.to(p.grad.device, dtype=p.grad.dtype))
                    offset += n

                # Clip gradients to avoid explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                optimizer.step()
                local_steps += 1

                epoch_loss += loss.item()
                preds = outputs.argmax(1)
                correct += (preds == yb).sum().item()
                total += yb.numel()

            avg_loss = epoch_loss / max(total, 1)
            acc = correct / max(total, 1)
            print(f"     [Client {self.id}] | Epoch {k+1}/{self.local_epochs} | Loss={avg_loss:.4f} | Acc={acc:.4f}")

        # --- Get final state (θ_{K}) and compute delta (Δθ = θ_{K} − θ_{0}) ---
        self.thetaK = self.params_to_vector(self.model)
        self.delta_state = self.thetaK - self.theta0

        # --- Update local control variate: c_i+ = c_i - c + (1/(τ η)) (θ^0 - θ^K) ---
        tau = max(local_steps, 1)    # total local steps
        ci_new = self.ci - self.c_global + (self.theta0 - self.thetaK) / (tau * float(self.local_learning_rate))

        self.delta_ci = ci_new - self.ci
        self.ci = ci_new.clone()                # update c_local


