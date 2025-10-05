import torch
import numpy as np
import copy

from torch.utils.data import DataLoader
from utils.utils_dataset import MultiDataset

class BaseServer(object):
    def __init__(self, args):
        self.seed = args.seed
        self.device = args.device
        self.global_model = copy.deepcopy(args.global_model).to(self.device)
        self.data_obj = args.data_obj        
        self.global_rounds = args.global_rounds
        self.global_learning_rate = args.global_learning_rate

        self.n_clients = args.n_clients
        self.selected_ratio = args.selected_ratio
        self.clients = []
        self.w_decay = args.weight_decay

        self.dataset_name = args.dataset_name

        self.test_acc_hist = []
        self.train_loss_hist = []

    @torch.no_grad()
    def params_to_vector(self, model):
        params = list(model.parameters())
        if not params:
            return torch.empty(0)
        return torch.cat([p.detach().view(-1).cpu() for p in params])


    @torch.no_grad()
    def vector_to_params(self, model, flat):
        params = list(model.parameters())
        if not params:
            return model

        flat = torch.as_tensor(flat) 
        offset = 0
        for p in params:
            n = p.numel()
            p.data.copy_(flat[offset:offset+n].view_as(p).to(p.device, dtype=p.dtype))
            offset += n

        if offset != flat.numel():
            raise ValueError(f"Size mismatch: got {flat.numel()}, expected {offset}")
        return model
        
    def setup_clients(self, args, clientObj):
        for id in range(self.n_clients):
            train_x = self.data_obj.clnt_x[id]
            train_y = self.data_obj.clnt_y[id]
            client = clientObj(args, id=id, train_x=train_x, 
                                            train_y=train_y)
            self.clients.append(client)

    def select_clients(self, epoch, seed=None):
        np.random.seed(epoch + seed)

        n_selected = int(self.n_clients * self.selected_ratio)
        selected = np.random.choice(self.n_clients, n_selected, replace=False)
        selected_clients = sorted(selected.tolist()) 
        print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clients])))
        return selected_clients  

    def send_model(self, client_id, model):        
        self.clients[client_id].model = copy.deepcopy(model)
        print(f"Send Current Global Model to Client {client_id}")
        
    def receive_models(self):        
        assert (len(self.selected_clients) > 0)
        self.uploaded_weights = []   #num of samples
        self.model_update = []
        total_samples = 0
        for client_id in self.selected_clients:
            cl = self.clients[client_id]
            total_samples += cl.num_samples
            self.uploaded_weights.append(cl.num_samples)
            self.model_update.append(cl.delta_state)

        self.uploaded_weights = [w / total_samples for w in self.uploaded_weights]


    def aggregate_parameters(self):
        """
        Returns:
            agg_delta: dict[name -> Tensor]  (global Δθ = Σ_i w_i * Δθ_i)
        """
        assert len(self.model_update) > 0, "No client updates."
        assert len(self.uploaded_weights) == len(self.model_update)

        with torch.no_grad():
            w = torch.tensor(self.uploaded_weights, dtype=torch.float32)
            agg = None
            for wi, di in zip(w, self.model_update):
                term = float(wi) * di.detach().cpu()
                agg = term if agg is None else agg + term
            return agg  # 1-D tensor


    @torch.no_grad()
    def update_global_model(self, agg_delta):
        """
        Returns the updated global model (also updated in-place).
        """
        # Make sure agg_delta is a tensor and on the same device as the flat vector we build
        if isinstance(agg_delta, np.ndarray):
            agg_delta = torch.from_numpy(agg_delta)

        cur_vec = self.params_to_vector(self.global_model)       
        new_global_ml = cur_vec + float(self.global_learning_rate) * agg_delta.to(cur_vec.dtype)
        self.global_model = self.vector_to_params(self.global_model, new_global_ml)

    def empty_cache(self):
        for client_id in self.selected_clients:
            del self.clients[client_id].model
            self.clients[client_id].model = None

    def get_acc_loss(self, data_x, data_y, model, w_decay=None):
        loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        batch_size = min(6000, data_x.shape[0])
        model.eval()
        model.to(self.device)

        # your required DataLoader line
        tst_gen = DataLoader(
            MultiDataset(data_x, data_y, dataset_name=self.dataset_name),
            batch_size=batch_size,
            shuffle=False)
        
        total_loss = 0.0
        total_correct = 0
        total = 0
        for xb, yb in tst_gen:
            xb = xb.to(self.device).float()
            yb = yb.to(self.device).long().view(-1)

            logits = model(xb)
            loss = loss_fn(logits, yb)            
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            total_correct += (preds == yb).sum().item()
            total += yb.numel()

        # mean loss over all samples
        avg_loss = total_loss / max(total, 1)

        # optional L2 regularization (added once)
        if w_decay is not None and float(w_decay) > 0.0:
            l2 = 0.0
            for p in model.parameters():
                if p.requires_grad:
                    l2 += p.detach().pow(2).sum().item()
            avg_loss = avg_loss + 0.5 * float(w_decay) * l2
        else:
            avg_loss = avg_loss
        acc = total_correct / max(total, 1)
        return avg_loss, acc

            




        



        



