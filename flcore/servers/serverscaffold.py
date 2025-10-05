import random
import torch
import numpy as np
import copy
import os
from flcore.clients.clientscaffold import clientScaffold
from flcore.servers.serverbase import BaseServer


class ServerSCAFFOLD(BaseServer):
    def __init__(self, args, verbose=True):
        super().__init__(args)

        # at server init, global control variate is created
        self.c_global = torch.zeros_like(self.params_to_vector(self.global_model), 
                                                    dtype=torch.float32,device="cpu")

        self.setup_clients(args, clientScaffold)
        print("Finished creating server and clients.")    

    # ---------------- Function ----------------

    def send_model(self, client_id, model):        
        self.clients[client_id].model = copy.deepcopy(model)
        self.clients[client_id].receive_controls(self.c_global)
        print(f"Send Current Global Model and Global Control Variate to Client {client_id}")

    def receive_models(self):        
        assert (len(self.selected_clients) > 0)
        self.uploaded_weights = []   #num of samples
        self.model_update = []
        self.control_update = []
        total_samples = 0
        for client_id in self.selected_clients:
            cl = self.clients[client_id]
            total_samples += cl.num_samples
            self.uploaded_weights.append(cl.num_samples)
            self.model_update.append(cl.delta_state)
            self.control_update.append(cl.delta_ci)

        self.uploaded_weights = [w / total_samples for w in self.uploaded_weights]

    def aggregate_controls(self):
        """
        Weighted average of client control updates (Δc_i = c_i' - c_i) → flat agg delta-c.
        """
        assert len(self.control_update) > 0, "No client updates."
        assert len(self.uploaded_weights) == len(self.control_update)

        with torch.no_grad():
            w = torch.tensor(self.uploaded_weights, dtype=torch.float32)
            agg = None
            for wi, dci in zip(w, self.control_update):
                term = float(wi) * dci.detach().cpu()
                agg = term if agg is None else agg + term
            return agg  # flat CPU tensor
        
    @torch.no_grad() 
    def update_global_control(self, agg_delta_c):
        """
        c ← c + Δc_global  (flat 1-D tensor, typically weighted average)
        """
        self.c_global = (self.c_global + agg_delta_c.to(self.c_global.dtype)).clone()


    # ------------------------------------------

    def train(self):

        if os.path.exists(f'Output/plot/test_acc_hist_SCAFFOLD_{self.global_rounds}_{self.dataset_name}_{self.data_obj.rule}_{self.data_obj.rule_arg}.npy'):
            print(f"SCAFFOLD is already trained on {self.dataset_name} {self.data_obj.rule} {self.data_obj.rule_arg}")
            self.test_acc_hist = []
            self.train_loss_hist = []

            self.test_acc_hist = np.load(f'Output/plot/test_acc_hist_SCAFFOLD_{self.global_rounds}_{self.dataset_name}_{self.data_obj.rule}_{self.data_obj.rule_arg}.npy')
            self.train_loss_hist = np.load(f'Output/plot/train_loss_hist_SCAFFOLD_{self.global_rounds}_{self.dataset_name}_{self.data_obj.rule}_{self.data_obj.rule_arg}.npy')
        else:
            # --------------- Training ---------------
            for epoch in range(self.global_rounds): 
                print(f"\n---------------------------Round: {epoch+1}---------------------------")
                self.selected_clients = self.select_clients(epoch, self.seed)

                for client_id in self.selected_clients:
                    self.send_model(client_id, self.global_model)
                    self.clients[client_id].train(epoch)  # param "epoch" is for lr decay 

                self.receive_models()
                agg_delta = self.aggregate_parameters()
                agg_delta_c = self.aggregate_controls()

                self.update_global_model(agg_delta)
                self.update_global_control(agg_delta_c)  
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
                
            np.save(f"Output/plot/test_acc_hist_SCAFFOLD_{self.global_rounds}_{self.dataset_name}_{self.data_obj.rule}_{self.data_obj.rule_arg}.npy", np.asarray(self.test_acc_hist, dtype=float))
            np.save(f"Output/plot/train_loss_hist_SCAFFOLD_{self.global_rounds}_{self.dataset_name}_{self.data_obj.rule}_{self.data_obj.rule_arg}.npy", np.asarray(self.train_loss_hist, dtype=float))

            