import time
import torch
import numpy as np
import copy
import os
from flcore.clients.clientfedgf import clientFedGF
from flcore.servers.serverbase import BaseServer


class ServerFedGF(BaseServer):
    def __init__(self, args, verbose=True):
        super().__init__(args)

        self.rho = args.fedsam_rho
        self.c_val = args.c_value           
           
        self.global_perturb = None

        self.setup_clients(args, clientFedGF)
        print("Finished creating server and clients.")

    def send_infor(self, client_id, model, global_perturb):
        self.clients[client_id].model = copy.deepcopy(model)
        if global_perturb is None:
            w_t_vec = self.params_to_vector(self.global_model).detach().cpu()
            global_perturb = w_t_vec.clone()
        self.clients[client_id].global_perturb = torch.as_tensor(global_perturb, dtype=torch.float32, device="cpu").clone()
        print(f"Send Current Global Model, Global Perturbation to Client {client_id}")

    
    @torch.no_grad()
    def update_global_model(self, agg_delta):
        """
        Returns the updated global model (also updated in-place).
        """
        # Make sure agg_delta is a tensor and on the same device as the flat vector we build
        if isinstance(agg_delta, np.ndarray):
            agg_delta = torch.from_numpy(agg_delta)

        cur_vec = self.params_to_vector(self.global_model)       
        new_global_ml = cur_vec - float(self.global_learning_rate) * agg_delta.to(cur_vec.dtype)
        self.global_model = self.vector_to_params(self.global_model, new_global_ml)


    @torch.no_grad()        
    def update_global_perturb(self, agg_delta):
        """
        Build the global perturbation for the next round:
            Δ^r      = mean_i (Δ_i^r)          # from self.model_update
            w_tilde^{r+1} = w^{r+1} + ρ * Δ^r / ||Δ^r||
        Stores CPU-flat vector in self.global_perturb and returns it.
        """
        # Safety: make sure we have a vectorized current global model
        w_t_vec = self.params_to_vector(self.global_model).detach().cpu()

        # normalize and build w_tilde^{r+1}
        d = agg_delta.detach().cpu()   
        dnorm = d.norm(p=2)
        if dnorm.item() > 0:
            self.global_perturb = w_t_vec + (self.rho / (dnorm + 1e-12)) * d
        else:
            self.global_perturb = w_t_vec.clone()


    def train(self):
        if os.path.exists(f'Output/plot/test_acc_hist_FedGF_{self.global_rounds}_{self.dataset_name}_{self.data_obj.rule}_{self.data_obj.rule_arg}.npy'):
            print(f"FedGF is already trained on {self.dataset_name} {self.data_obj.rule} {self.data_obj.rule_arg}")
            self.test_acc_hist = []
            self.train_loss_hist = []

            self.test_acc_hist = np.load(f'Output/plot/test_acc_hist_FedGF_{self.global_rounds}_{self.dataset_name}_{self.data_obj.rule}_{self.data_obj.rule_arg}.npy')
            self.train_loss_hist = np.load(f'Output/plot/train_loss_hist_FedGF_{self.global_rounds}_{self.dataset_name}_{self.data_obj.rule}_{self.data_obj.rule_arg}.npy')
        else:
            # --------------- Training ---------------
            for epoch in range(self.global_rounds):
                print(f"\n---------------------------Round: {epoch+1}---------------------------")
                self.selected_clients = self.select_clients(epoch, self.seed)

                for client_id in self.selected_clients:
                    self.send_infor(client_id, self.global_model, self.global_perturb)
                    self.clients[client_id].train(epoch)       # param "epoch" is for lr decay

                self.receive_models()
                agg_delta = self.aggregate_parameters()

                self.update_global_model(agg_delta)
                self.update_global_perturb(agg_delta)

                self.empty_cache()

                #### Get Loss and Acc
                loss_tst, acc_tst = self.get_acc_loss(self.data_obj.tst_x, self.data_obj.tst_y, 
                                                model=self.global_model, w_decay=self.w_decay)
                self.test_acc_hist.append(acc_tst) 

                loss_train, acc_train = self.get_acc_loss(np.concatenate(self.data_obj.clnt_x), 
                                                        np.concatenate(self.data_obj.clnt_y), 
                                                        model=self.global_model, 
                                                        w_decay=self.w_decay)
                self.train_loss_hist.append(loss_train)

                print(
                    "**** Communication sel %3d, Cent Accuracy: %.4f, Cent Loss: %.4f, Test Accuracy: %.4f, Test Loss: %.4f"
                    % (epoch+1, acc_train, loss_train, acc_tst, loss_tst))

            np.save(f"Output/plot/test_acc_hist_FedGF_{self.global_rounds}_{self.dataset_name}_{self.data_obj.rule}_{self.data_obj.rule_arg}.npy", np.asarray(self.test_acc_hist, dtype=float))
            np.save(f"Output/plot/train_loss_hist_FedGF_{self.global_rounds}_{self.dataset_name}_{self.data_obj.rule}_{self.data_obj.rule_arg}.npy", np.asarray(self.train_loss_hist, dtype=float))



