import torch
from flcore.clients.clientbase import Client


class clientFedDyn(Client):
    def __init__(self, args, id, train_x, train_y, **kwargs):
        super().__init__(args, id, train_x, train_y, **kwargs)
        # β for FedDyn
        self.beta = float(args.feddyn_beta)
        self.lamb_i = None

    def train(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.local_learning_rate,
                                    weight_decay=self.weight_decay,momentum=self.momentum)
        self.trainloader = self.load_train_data()
        
        self.model.train()
        self.model = self.model.to(self.device)

        # --- get initial state (θ^{0}) ---
        self.theta0 = self.params_to_vector(self.model)
        # --- get current lambda (λ_i^t) ---
        lamb_cpu = self.lamb_i if self.lamb_i is not None else torch.zeros_like(self.theta0)

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

                # ---- FedDyn dynamic correction: g <- g + ((1/β)(w − w^t) - λ_i^t) ----
                offset = 0
                # Chuẩn bị lát của w^t (theta0) & λ_i^t theo thứ tự tham số
                for p in self.model.parameters():
                    n = p.numel()
                    if p.grad is not None:
                        # (-λ) term
                        lam_slice = (-lamb_cpu[offset:offset+n]).view_as(p).to(p.grad.device, dtype=p.grad.dtype)
                        p.grad.add_(lam_slice)
                        # proximal term (1/β)(w - w^t)
                        w0_slice = self.theta0[offset:offset+n].view_as(p).to(p.grad.device, dtype=p.grad.dtype)
                        prox = (1.0 / self.beta) * (p.data - w0_slice)
                        p.grad.add_(prox)
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

        # --- Update local dual control variate: λ_i+ = λ_i - (1/β) (θ^K - θ^0) ---
        lamb_new = lamb_cpu - (1.0 / self.beta) * self.delta_state
        self.lamb_i = lamb_new.clone()  # cache local


       
