"""
Client implementation for FedAvg.
Assumes each client's data is provided as numpy arrays (X: N,C,H,W,  y: N,1 or N).
Converts to torch.Tensor and uses a simple DataLoader for local training.
"""
from flcore.clients.clientbase import Client
import torch



class clientAvg(Client):
    def __init__(self, args, id, train_x, train_y, **kwargs):
        super().__init__(args, id, train_x, train_y, **kwargs)

    
    def train(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.local_learning_rate,
                                    weight_decay=self.weight_decay,momentum=self.momentum)
        self.trainloader = self.load_train_data()

        self.model.train()
        self.model = self.model.to(self.device)

        # --- get initial state (θ^{0}) ---
        self.theta0 = self.params_to_vector(self.model)

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




        