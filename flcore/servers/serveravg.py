import time
import numpy as np
import os
from flcore.clients.clientavg import clientAvg
from flcore.servers.serverbase import BaseServer


class ServerAvg(BaseServer):
    def __init__(self, args, verbose=True):
        super().__init__(args)

        self.setup_clients(args, clientAvg)
        print("Finished creating server and clients.")


    def train(self):       

        if os.path.exists(f'Output/plot/test_acc_hist_FedAvg_{self.global_rounds}_{self.dataset_name}_{self.data_obj.rule}_{self.data_obj.rule_arg}.npy'):
            print(f"FedAvg is already trained on {self.dataset_name} {self.data_obj.rule} {self.data_obj.rule_arg}")
            self.test_acc_hist = []
            self.train_loss_hist = []

            self.test_acc_hist = np.load(f'Output/plot/test_acc_hist_FedAvg_{self.global_rounds}_{self.dataset_name}_{self.data_obj.rule}_{self.data_obj.rule_arg}.npy')
            self.train_loss_hist = np.load(f'Output/plot/train_loss_hist_FedAvg_{self.global_rounds}_{self.dataset_name}_{self.data_obj.rule}_{self.data_obj.rule_arg}.npy')
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

            np.save(f"Output/plot/test_acc_hist_FedAvg_{self.global_rounds}_{self.dataset_name}_{self.data_obj.rule}_{self.data_obj.rule_arg}.npy", np.asarray(self.test_acc_hist, dtype=float))
            np.save(f"Output/plot/train_loss_hist_FedAvg_{self.global_rounds}_{self.dataset_name}_{self.data_obj.rule}_{self.data_obj.rule_arg}.npy", np.asarray(self.train_loss_hist, dtype=float))
            