import torch
from flcore.clients.clientbase import Client


class clientFedDyn(Client):
    def __init__(self, args, id, train_x, train_y, **kwargs):
        super().__init__(args, id, train_x, train_y, **kwargs)
        # β for FedDyn
        self.beta = float(args.feddyn_beta)
        self.lamb_i = None

    def train(self, global_round):
        self.model.train()
        self.model = self.model.to(self.device)
        # lr decay: .9995 only for FedDyn
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.local_learning_rate * (float(0.9995) ** global_round),
                                    weight_decay=self.weight_decay,momentum=self.momentum)
        self.trainloader = self.load_train_data()

        # --- get initial state (w_i_0^r) ---
        self.theta0 = self.params_to_vector(self.model).detach()
        # --- get current lambda (λ_i^r) ---
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

                # ---- FedDyn dynamic correction: g <- g + ((1/β)(w − w^r) - λ_i^r) ----
                offset = 0
                for p in self.model.parameters():
                    n = p.numel()
                    if p.grad is not None:
                        # (-λ_i^r) term
                        lam_slice = (-lamb_cpu[offset:offset+n]).view_as(p).to(p.grad.device, dtype=p.grad.dtype)
                        p.grad.add_(lam_slice)
                        # proximal term (1/β)(w - w^r)
                        w0_slice = self.theta0[offset:offset+n].view_as(p).to(p.grad.device, dtype=p.grad.dtype)
                        prox = (1.0 / self.beta) * (p.data - w0_slice)
                        p.grad.add_(prox)
                    offset += n

                # Clip gradients to avoid explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                optimizer.step()
                local_steps += 1

                epoch_loss += loss.item() * yb.numel()
                preds = outputs.argmax(1)
                correct += (preds == yb).sum().item()
                total += yb.numel()

            avg_loss = epoch_loss / max(total, 1)
            acc = correct / max(total, 1)
            print(f"     [Client {self.id}] | Epoch {k+1}/{self.local_epochs} | Loss={avg_loss:.4f} | Acc={acc:.4f}")

        # --- Get final state (w_i_K^r) and compute delta (Δw_i^r = w_i_K^r − w^r) ---
        self.thetaK = self.params_to_vector(self.model).detach()
        self.delta_state = self.thetaK - self.theta0

        # --- Update local dual control variate: λ_i^{r+1} = λ_i^r - (1/β) (w_i_K^r − w^r) ---
        lamb_new = lamb_cpu - (1.0 / self.beta) * self.delta_state
        self.lamb_i = lamb_new.clone()  # cache local


       
