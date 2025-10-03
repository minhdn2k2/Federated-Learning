import torch
import argparse
import warnings
import numpy as np
import logging
import random
import os
import matplotlib.pyplot as plt

from utils.utils_dataset import DatasetObject
from utils.utils_model import CNN

from flcore.servers.serveravg import ServerAvg
from flcore.servers.serverscaffold import ServerSCAFFOLD
from flcore.servers.serverfeddyn import ServerFedDyn


logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('-data', "--dataset_name", type=str, default="CIFAR10")
    parser.add_argument('-nc', "--n_clients", type=int, default=100,help="Total number of clients")
    parser.add_argument("--rule", type=str, default='Dirichlet')
    parser.add_argument("--rule_arg", type=float, default=0.3)
    parser.add_argument("--unbalanced_sgm", type=float, default=0.0)

    # Hyperparam of FL system
    parser.add_argument('-dev', "--device", type=str, default="cuda")
    parser.add_argument('-gr', "--global_rounds", type=int, default=500)
    parser.add_argument('-lbs', "--batch_size", type=int, default=50)
    parser.add_argument("--global_learning_rate", type=float, default=1.0)

    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.01)
    parser.add_argument('--lr_decay', type=float, default=0.998)
    parser.add_argument('--weight_decay', type=float, default=0.0) #0.1e-5
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--global_momentum', type=float, default=0.9) # For FedAvgM

    parser.add_argument('-le', "--local_epochs", type=int, default=10)

    parser.add_argument('-jr', "--selected_ratio", type=float, default=0.1,help="Ratio of clients per round")

    #FedDyn
    parser.add_argument("--feddyn_beta", type=float, default=10)

    args = parser.parse_args()

    torch.manual_seed(args.seed)  # cpu
    torch.cuda.manual_seed(args.seed)  # gpu
    np.random.seed(args.seed)  # numpy
    random.seed(args.seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn 
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


    args.data_obj = DatasetObject(dataset=args.dataset_name, n_client=args.n_clients, seed=args.seed, rule=args.rule, 
                             unbalanced_sgm=args.unbalanced_sgm, rule_arg=args.rule_arg)
    args.num_classes = args.data_obj.n_cls
    args.global_model = CNN(num_classes=args.num_classes).to(args.device)


    # Check and create if it does not exist
    if not os.path.exists("Output/plot"):
        os.makedirs("Output/plot")
        print("Folder Output/plot has been created.")
    else:
        print("Folder Output/plot already exists.")


    # FedAvg
    print("FedAvg")
    server_avg = ServerAvg(args=args)
    server_avg.train()
    fedavg_test_acc = np.asarray(server_avg.test_acc_hist, dtype=float)
    fedavg_train_loss = np.asarray(server_avg.train_loss_hist, dtype=float)

    # SCAFFOLD
    print('SCAFFOLD')
    server_SCAFFOLD = ServerSCAFFOLD(args=args)
    server_SCAFFOLD.train()
    SCAFFOLD_test_acc = np.asarray(server_SCAFFOLD.test_acc_hist, dtype=float)
    SCAFFOLD_train_loss = np.asarray(server_SCAFFOLD.train_loss_hist, dtype=float)


    # FedDyn
    print('FedDyn')
    server_FedDyn = ServerFedDyn(args=args)
    server_FedDyn.train()
    FedDyn_test_acc = np.asarray(server_FedDyn.test_acc_hist, dtype=float)
    FedDyn_train_loss = np.asarray(server_FedDyn.train_loss_hist, dtype=float)


    # Plot 1: Test Accuracy vs Rounds
    plt.figure(figsize=(6, 5))
    plt.plot(np.arange(1, args.global_rounds + 1), fedavg_test_acc, label="FedAvg", color="#0072B2", linewidth=1) 
    plt.plot(np.arange(1, args.global_rounds + 1), SCAFFOLD_test_acc, label="SCAFFOLD", color="#41B200", linewidth=1)
    plt.plot(np.arange(1, args.global_rounds + 1), FedDyn_test_acc, label="FedDyn", color="#B2A900", linewidth=1)
    plt.xlabel("Global Round", fontsize=16)
    plt.ylabel("Test Accuracy", fontsize=16)
    plt.legend(fontsize=16, loc='lower right', bbox_to_anchor=(1.015, -0.02))
    plt.grid()
    plt.xlim([0, args.global_rounds + 1])
    plt.title(f'{args.dataset_name} {args.rule} {args.rule_arg}', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(f'Output/Test_Accuracy_{args.dataset_name}_{args.rule}_{args.rule_arg}.png', dpi=1000, bbox_inches='tight')

    # Plot 2: Training Loss vs Rounds
    plt.figure(figsize=(6, 5))
    plt.plot(np.arange(1, args.global_rounds + 1), fedavg_train_loss, label="FedAvg", color="#0072B2", linewidth=1)
    plt.plot(np.arange(1, args.global_rounds + 1), SCAFFOLD_train_loss, label="SCAFFOLD", color="#41B200", linewidth=1)
    plt.plot(np.arange(1, args.global_rounds + 1), FedDyn_train_loss, label="FedDyn", color="#B2A900", linewidth=1)
    plt.xlabel("Global Round", fontsize=16)
    plt.ylabel("Train Loss", fontsize=16)
    plt.legend(fontsize=16, loc='upper right', bbox_to_anchor=(1.015, 1.02))
    plt.grid()
    plt.xlim([0, args.global_rounds + 1])
    plt.title(f'{args.dataset_name} {args.rule} {args.rule_arg}', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(f'Output/Train_Loss_{args.dataset_name}_{args.rule}_{args.rule_arg}.png', dpi=1000, bbox_inches='tight')

