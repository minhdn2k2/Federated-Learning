from flcore.clients.clientbase import Client
import torch


class clientAvg(Client):
    def __init__(self, args, id, train_x, train_y, **kwargs):
        super().__init__(args, id, train_x, train_y, **kwargs)

    
    def train(self, global_round):
        self.model.train()
        self.model = self.model.to(self.device)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.local_learning_rate * (self.lr_decay ** global_round),
                                    weight_decay=self.weight_decay,momentum=self.momentum)
        self.trainloader = self.load_train_data()

        # --- get initial state (w_i_0^r) ---
        self.theta0 = self.params_to_vector(self.model).detach()

        for k in range(self.local_epochs):
            epoch_loss, correct, total = 0.0, 0, 0
            for xb, yb in self.trainloader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True).long().view(-1)

                outputs = self.model(xb)
                loss = self.loss(outputs, yb)
                optimizer.zero_grad()
                loss.backward()
                # Clip gradients to avoid explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                optimizer.step()

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




        