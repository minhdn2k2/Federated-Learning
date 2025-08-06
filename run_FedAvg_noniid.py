from utils_dataset import *
from utils_general import *
from utils_methods import *
from utils_models import *


# Dataset initialization

########
# For 'CIFAR100' experiments
#     - Change the dataset argument from CIFAR10 to CIFAR100.
########
# For 'mnist' experiments
#     - Change the dataset argument from CIFAR10 to mnist.
########
# For 'emnist' experiments
#     - Download emnist dataset from (https://www.nist.gov/itl/products-and-services/emnist-dataset) as matlab format and unzip it in "Data/Raw/" folder.
#     - Change the dataset argument from CIFAR10 to emnist.
########
# For Shakespeare experiments
# First generate dataset using LEAF Framework and set storage_path to the data folder
# storage_path = 'LEAF/shakespeare/data/'
#     - In IID use

# name = 'shakepeare'
# data_obj = ShakespeareObjectCrop(storage_path, dataset_prefix)

#     - In non-IID use
# name = 'shakepeare_nonIID'
# data_obj = ShakespeareObjectCrop_noniid(storage_path, dataset_prefix)
#########


# Generate IID or Dirichlet distribution
# IID
n_client = 100
data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, rule='Dirichlet', unbalanced_sgm=0.3, rule_arg=0.3)

###
model_name         = 'cifar10'     # Model type
com_amount         = 100          # Global round
save_period        = 50
weight_decay       = 1e-3
batch_size         = 50
# act_prob           = 1
praction_clnt      = 0.1
lr_decay_per_round = 1
epoch              = 5             # Local epoch   
learning_rate      = 0.1
print_per          = 5


model_func = lambda : client_model(model_name)
init_model = model_func()

if not os.path.exists('Output/%s/%s_init_mdl.pt' %(data_obj.name, model_name)):
    print("New directory!")
    os.mkdir('Output/%s/' %(data_obj.name))
    torch.save(init_model.state_dict(), 'Output/%s/%s_init_mdl.pt' %(data_obj.name, model_name))
else:
    # Load model
    init_model.load_state_dict(torch.load('Output/%s/%s_init_mdl.pt' %(data_obj.name, model_name)))



print('FedAvg')

[fed_mdls_sel_FedAvg, trn_perf_sel_FedAvg, 
 tst_perf_sel_FedAvg] = train_FedAvg(data_obj=data_obj, praction_clnt=praction_clnt, learning_rate=learning_rate, batch_size=batch_size,
                                     epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
                                     model_func=model_func, init_model=init_model, save_period=save_period,
                                     lr_decay_per_round=lr_decay_per_round)

# Plot results
plt.figure(figsize=(6, 5))
plt.plot(np.arange(com_amount)+1, trn_perf_sel_FedAvg[:,0], label='FedAvg')
plt.ylabel('Train Loss', fontsize=16)
plt.xlabel('Communication Rounds', fontsize=16)
plt.legend(fontsize=16, loc='lower right', bbox_to_anchor=(1.015, -0.02))
plt.grid()
plt.xlim([0, com_amount+1])
plt.title(data_obj.name, fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('Output/%s/Train Loss FedAvg.pdf' %data_obj.name, dpi=1000, bbox_inches='tight')
# plt.show() 

plt.figure(figsize=(6, 5))
plt.plot(np.arange(com_amount)+1, tst_perf_sel_FedAvg[:,1], label='FedAvg')
plt.ylabel('Test Accuracy', fontsize=16)
plt.xlabel('Communication Rounds', fontsize=16)
plt.legend(fontsize=16, loc='lower right', bbox_to_anchor=(1.015, -0.02))
plt.grid()
plt.xlim([0, com_amount+1])
plt.title(data_obj.name, fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('Output/%s/Test Accuracy FedAvg.pdf' %data_obj.name, dpi=1000, bbox_inches='tight')







