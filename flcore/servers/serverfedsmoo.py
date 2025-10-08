import time
from flcore.clients.clientfedsmoo import clientFedSMOO
from flcore.servers.serverbase import BaseServer

import numpy as np
import torch
import os
import copy


class ServerFedSMOO(BaseServer):
    def __init__(self, args, verbose=True):
        super().__init__(args)

        self.rho = args.fedsam_rho
        self.beta =  args.beta_fedsmoo

        # global dual and global perturbation (flatten, CPU)
        self.lambda_g = None   # λ^t
        self.global_s = None   # s^t

        self.setup_clients(args, clientFedSMOO)
        print("Finished creating server and clients.")

    # ---------------- Function -----------------

    def send_model(self, client_id, model):
        self.clients[client_id].model = copy.deepcopy(model)
        
        if self.global_s is None:
            self.global_s = torch.zeros_like(self.params_to_vector(self.global_model), 
                                                        dtype=torch.float32,device="cpu")
        self.clients[client_id].get_global_s(self.global_s)
        print(f"Send Current Global Model and Global Perturbation to Client {client_id}")

    def receive_models(self):
        assert (len(self.selected_clients) > 0)
        self.uploaded_weights = []   #num of samples
        self.updated_model = []
        self.s_e_list = []
        self.deltas_list = []
        total_samples = 0
        for client_id in self.selected_clients:
            cl = self.clients[client_id]
            total_samples += cl.num_samples
            self.uploaded_weights.append(cl.num_samples)
            self.updated_model.append(cl.thetaK)
            self.s_e_list.append(cl.s_e.detach().cpu())
            self.deltas_list.append(cl.delta_state.detach().cpu())

        self.uploaded_weights = [w / total_samples for w in self.uploaded_weights]

    def aggregate_parameters(self):
        assert len(self.updated_model) > 0, "No client updates."
        assert len(self.uploaded_weights) == len(self.updated_model)

        with torch.no_grad():
            w = torch.tensor(self.uploaded_weights, dtype=torch.float32)
            agg = None
            for wi, di in zip(w, self.updated_model):
                term = float(wi) * di.detach().cpu()
                agg = term if agg is None else agg + term
            return agg  # 1-D tensor



    def update_gl_perturb(self):
        s_tmp = None
        for se in self.s_e_list:
            s_tmp = se.clone() if s_tmp is None else s_tmp.add_(se)
        s_tmp.div_(len(self.s_e_list))
        norm = s_tmp.norm(p=2)

        if norm.item() > 0:
            self.global_s = (self.rho / (norm + 1e-12)) * s_tmp
        else:
            self.global_s = torch.zeros_like(s_tmp)

    def update_gl_dual(self):
        if self.lambda_g is None:
            self.lambda_g = torch.zeros_like(self.params_to_vector(self.global_model), 
                                                        dtype=torch.float32,device="cpu")
        sum_delta = None
        for d in self.deltas_list:
            sum_delta = d.clone() if sum_delta is None else sum_delta.add_(d)
        m = len(self.clients) if len(self.clients) > 0 else 1
        self.lambda_g.add_( - (1.0 / (self.beta * float(m))) * sum_delta )

    def update_global_model_fedsmoo(self, mean_w):
        global_model = mean_w - self.beta * self.lambda_g
        self.global_model = self.vector_to_params(self.global_model, global_model)




    def train(self):
        if os.path.exists(f'Output/plot/test_acc_hist_FedSMOO_{self.global_rounds}_{self.dataset_name}_{self.data_obj.rule}_{self.data_obj.rule_arg}.npy'):
            print(f"FedSMOO is already trained on {self.dataset_name} {self.data_obj.rule} {self.data_obj.rule_arg}")
            self.test_acc_hist = []
            self.train_loss_hist = []

            self.test_acc_hist = np.load(f'Output/plot/test_acc_hist_FedSMOO_{self.global_rounds}_{self.dataset_name}_{self.data_obj.rule}_{self.data_obj.rule_arg}.npy')
            self.train_loss_hist = np.load(f'Output/plot/train_loss_hist_FedSMOO_{self.global_rounds}_{self.dataset_name}_{self.data_obj.rule}_{self.data_obj.rule_arg}.npy')
        else:
            # --------------- Training ---------------
            for epoch in range(self.global_rounds): 
                print(f"\n---------------------------Round: {epoch+1}---------------------------")

                self.selected_clients = self.select_clients(epoch, self.seed)

                for client_id in self.selected_clients:
                    self.send_model(client_id, self.global_model)
                    self.clients[client_id].train(epoch)

                self.receive_models()
                agg_client_ml = self.aggregate_parameters()   # (1/n) sum w_i  -> mean w_i_k^r
                self.update_gl_perturb()        # s^{r+1} = p * normalize( mean s_e_i )
                self.update_gl_dual()           # λ^{r+1} = λ^r - (1/(β n)) * sum_i Δw_i
                self.update_global_model_fedsmoo(agg_client_ml)

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

            np.save(f"Output/plot/test_acc_hist_FedSMOO_{self.global_rounds}_{self.dataset_name}_{self.data_obj.rule}_{self.data_obj.rule_arg}.npy", np.asarray(self.test_acc_hist, dtype=float))
            np.save(f"Output/plot/train_loss_hist_FedSMOO_{self.global_rounds}_{self.dataset_name}_{self.data_obj.rule}_{self.data_obj.rule_arg}.npy", np.asarray(self.train_loss_hist, dtype=float))



