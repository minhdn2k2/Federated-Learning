import time
import numpy as np
import os
import copy
import torch
from tqdm import tqdm

from flcore.clients.clientmofedsam import clientMoFedSAM
from flcore.servers.serverbase import BaseServer


class ServerMoFedSAM(BaseServer):
    def __init__(self, args, verbose=True):
        super().__init__(args)

        self.local_learning_rate = args.local_learning_rate
        self.prev_global_update = None
        self.beta = args.beta_mofedsam

        self.setup_clients(args, clientMoFedSAM)
        print("Finished creating server and clients.")

    # ---------------- Function -----------------

    def send_model(self, client_id, model):        
        self.clients[client_id].model = copy.deepcopy(model)
        self.clients[client_id].receive_prev_gl_update(self.prev_global_update)
        print(f"Send Current Global Model and Previous Global Update to Client {client_id}")

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
            self.K = cl.K

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
            return agg #/ (self.K * self.local_learning_rate)  # 1-D tensor    

    # -------------------------------------------

    def train(self):       

        if os.path.exists(f'Output/plot/test_acc_hist_MoFedSAM_{self.beta}_{self.global_rounds}_{self.dataset_name}_{self.data_obj.rule}_{self.data_obj.rule_arg}.npy'):
            print(f"MoFedSAM with beta {self.beta} is already trained on {self.dataset_name} {self.data_obj.rule} {self.data_obj.rule_arg}")
            self.test_acc_hist = []
            self.train_loss_hist = []

            self.test_acc_hist = np.load(f'Output/plot/test_acc_hist_MoFedSAM_{self.beta}_{self.global_rounds}_{self.dataset_name}_{self.data_obj.rule}_{self.data_obj.rule_arg}.npy')
            self.train_loss_hist = np.load(f'Output/plot/train_loss_hist_MoFedSAM_{self.beta}_{self.global_rounds}_{self.dataset_name}_{self.data_obj.rule}_{self.data_obj.rule_arg}.npy')
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
                self.prev_global_update = agg_delta      # Store global model update
                self.update_global_model(agg_delta)

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

            np.save(f"Output/plot/test_acc_hist_MoFedSAM_{self.beta}_{self.global_rounds}_{self.dataset_name}_{self.data_obj.rule}_{self.data_obj.rule_arg}.npy", np.asarray(self.test_acc_hist, dtype=float))
            np.save(f"Output/plot/train_loss_hist_MoFedSAM_{self.beta}_{self.global_rounds}_{self.dataset_name}_{self.data_obj.rule}_{self.data_obj.rule_arg}.npy", np.asarray(self.train_loss_hist, dtype=float))


