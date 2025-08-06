from utils_libs import *
from utils_dataset import *
from utils_models import *
from utils_general import *

### Methods
def train_FedAvg(data_obj, praction_clnt ,learning_rate, batch_size, epoch, com_amount, print_per, weight_decay, model_func, init_model, save_period, lr_decay_per_round, rand_seed=0):
    method_name = 'FedAvg'
    n_clnt=data_obj.n_client
    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y
    
    cent_x = np.concatenate(clnt_x, axis=0)
    cent_y = np.concatenate(clnt_y, axis=0)
    
    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list.reshape((n_clnt, 1))
    
    if not os.path.exists('Output/%s/%s' %(data_obj.name, method_name)):
        os.mkdir('Output/%s/%s' %(data_obj.name, method_name))
        
    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances)) #fed_mdls_all = list(range(n_save_instances))
    
    trn_perf_sel = np.zeros((com_amount, 2)) #trn_perf_all = np.zeros((com_amount, 2))
    tst_perf_sel = np.zeros((com_amount, 2)) #tst_perf_all = np.zeros((com_amount, 2))
    n_par = len(get_mdl_params([model_func()])[0])

    init_par_list=get_mdl_params([init_model], n_par)[0]
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    
    avg_model = model_func().to(device)
    avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

    # all_model = model_func().to(device)
    # all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    
    if os.path.exists('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, com_amount)):
        # Load performances and models...
        for j in range(n_save_instances):
            fed_model = model_func()
            fed_model.load_state_dict(torch.load('Output/%s/%s/%d_com_sel.pt' %(data_obj.name, method_name, (j+1)*save_period)))
            fed_model.eval()
            fed_model = fed_model.to(device)
            fed_mdls_sel[j] = fed_model
            
            # fed_model = model_func()
            # fed_model.load_state_dict(torch.load('Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (j+1)*save_period)))
            # fed_model.eval()
            # fed_model = fed_model.to(device)
            # fed_mdls_all[j] = fed_model
        
        trn_perf_sel = np.load('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, com_amount))
        # trn_perf_all = np.load('Output/%s/%s/%d_com_trn_perf_all.npy' %(data_obj.name, method_name, com_amount))

        tst_perf_sel = np.load('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, com_amount))
        # tst_perf_all = np.load('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, com_amount))
        
        clnt_params_list = np.load('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, com_amount))
        
    else:
        for i in range(com_amount):

            # inc_seed = 0
            # while(True):
            #     # Fix randomness in client selection
            #     np.random.seed(i + rand_seed + inc_seed)
            #     act_list    = np.random.uniform(size=n_clnt)
            #     act_clients = act_list <= act_prob
            #     selected_clnts = np.sort(np.where(act_clients)[0])
            #     inc_seed += 1
            #     if len(selected_clnts) != 0:
            #         break
            num_selected = int(n_clnt * praction_clnt)
            selected_clnts = sorted(random.sample(range(n_clnt), num_selected))
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))

            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))

                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                clnt_models[clnt] = train_model(clnt_models[clnt], trn_x, trn_y, learning_rate * (lr_decay_per_round ** i), batch_size, epoch, print_per, weight_decay, data_obj.dataset)

                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]

            # Scale with weights
            # Mean of selected clients's updated models
            avg_model = set_client_from_params(model_func(), np.sum(clnt_params_list[selected_clnts]*weight_list[selected_clnts]/np.sum(weight_list[selected_clnts]), axis = 0))
            # Mean of all client's updated models
            # all_model = set_client_from_params(model_func(), np.sum(clnt_params_list*weight_list/np.sum(weight_list), axis = 0))

            ### Evaluate on test set and avg_model
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, avg_model, data_obj.dataset)
            tst_perf_sel[i] = [loss_tst, acc_tst]
            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))
            ### Evaluate on all train set and avg_model
            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y, avg_model, data_obj.dataset)
            trn_perf_sel[i] = [loss_tst, acc_tst]
            print("**** Communication sel %3d, Cent Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))
            ### Evaluate on test set and all_model
            # loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, all_model, data_obj.dataset)
            # tst_perf_all[i] = [loss_tst, acc_tst]
            # print("**** Communication all %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))
            ### Evaluate on all train set and avg_model
            # loss_tst, acc_tst = get_acc_loss(cent_x, cent_y, all_model, data_obj.dataset)
            # trn_perf_all[i] = [loss_tst, acc_tst]
            # print("**** Communication all %3d, Cent Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))

            if ((i+1) % save_period == 0):
                torch.save(avg_model.state_dict(), 'Output/%s/%s/%d_com_sel.pt' %(data_obj.name, method_name, (i+1))) 
                # torch.save(all_model.state_dict(), 'Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (i+1)))
                np.save('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, (i+1)), clnt_params_list)

                np.save('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, (i+1)), trn_perf_sel[:i+1])
                np.save('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, (i+1)), tst_perf_sel[:i+1])

                # np.save('Output/%s/%s/%d_com_trn_perf_all.npy' %(data_obj.name, method_name, (i+1)), trn_perf_all[:i+1])
                # np.save('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, (i+1)), tst_perf_all[:i+1])

                if (i+1) > save_period:
                    if os.path.exists('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, i+1-save_period)):
                        # Delete the previous saved arrays
                        os.remove('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, i+1-save_period))
                        os.remove('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, i+1-save_period))

                        # os.remove('Output/%s/%s/%d_com_trn_perf_all.npy' %(data_obj.name, method_name, i+1-save_period))
                        # os.remove('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, i+1-save_period))

                        os.remove('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, i+1-save_period))

            if ((i+1) % save_period == 0):
                fed_mdls_sel[i//save_period] = avg_model
                # fed_mdls_all[i//save_period] = all_model
                
    return fed_mdls_sel, trn_perf_sel, tst_perf_sel

def train_FedProx(data_obj, praction_clnt ,learning_rate, batch_size, epoch, com_amount, print_per, weight_decay, model_func, init_model, save_period, mu, lr_decay_per_round, rand_seed=0):
    method_name = 'FedProx'

    n_clnt=data_obj.n_client
    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y
    
    cent_x = np.concatenate(clnt_x, axis=0)
    cent_y = np.concatenate(clnt_y, axis=0)

    # Average them based on number of datapoints (The one implemented)
    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list.reshape((n_clnt, 1))

    if not os.path.exists('Output/%s/%s' %(data_obj.name, method_name)):
        os.mkdir('Output/%s/%s' %(data_obj.name, method_name))

    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances))# fed_mdls_all = list(range(n_save_instances))
    
    trn_perf_sel = np.zeros((com_amount, 2))# trn_perf_all = np.zeros((com_amount, 2))
    tst_perf_sel = np.zeros((com_amount, 2))# tst_perf_all = np.zeros((com_amount, 2))
    n_par = len(get_mdl_params([model_func()])[0])

    init_par_list=get_mdl_params([init_model], n_par)[0]
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    
    avg_model = model_func().to(device)
    avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

    # all_model = model_func().to(device)
    # all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

    if os.path.exists('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, com_amount)):
        # Load performances and models...
        for j in range(n_save_instances):
            fed_model = model_func()
            fed_model.load_state_dict(torch.load('Output/%s/%s/%d_com_sel.pt' %(data_obj.name, method_name, (j+1)*save_period)))
            fed_model.eval()
            fed_model = fed_model.to(device)
            fed_mdls_sel[j] = fed_model
            
            # fed_model = model_func()
            # fed_model.load_state_dict(torch.load('Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (j+1)*save_period)))
            # fed_model.eval()
            # fed_model = fed_model.to(device)
            # fed_mdls_all[j] = fed_model

        trn_perf_sel = np.load('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, com_amount))
        # trn_perf_all = np.load('Output/%s/%s/%d_com_trn_perf_all.npy' %(data_obj.name, method_name, com_amount))

        tst_perf_sel = np.load('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, com_amount))
        # tst_perf_all = np.load('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, com_amount))
        
        clnt_params_list = np.load('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, com_amount))
    
    else:
        for i in range(com_amount):
            num_selected = int(n_clnt * praction_clnt)
            selected_clnts = sorted(random.sample(range(n_clnt), num_selected))
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))
            avg_model_param = get_mdl_params([avg_model], n_par)[0]
            avg_model_param_tensor = torch.tensor(avg_model_param, dtype=torch.float32, device=device)

            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]
                clnt_models[clnt] = model_func().to(device)
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))
                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                clnt_models[clnt] = train_fedprox_mdl(clnt_models[clnt], avg_model_param_tensor, mu, trn_x, trn_y, learning_rate * (lr_decay_per_round ** i), batch_size, epoch, print_per, weight_decay, data_obj.dataset)
                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]

                # Scale with weights
                avg_model = set_client_from_params(model_func(), np.sum(clnt_params_list[selected_clnts]*weight_list[selected_clnts]/np.sum(weight_list[selected_clnts]), axis = 0))

                ###
                loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, avg_model, data_obj.dataset)
                tst_perf_sel[i] = [loss_tst, acc_tst]
                print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))
                ###
                loss_tst, acc_tst = get_acc_loss(cent_x, cent_y, avg_model, data_obj.dataset)
                trn_perf_sel[i] = [loss_tst, acc_tst]
                print("**** Communication sel %3d, Cent Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))
                ###
                # loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, all_model, data_obj.dataset)
                # tst_perf_all[i] = [loss_tst, acc_tst]
                # print("**** Communication all %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))
                ###
                # loss_tst, acc_tst = get_acc_loss(cent_x, cent_y, all_model, data_obj.dataset)
                # trn_perf_all[i] = [loss_tst, acc_tst]
                # print("**** Communication all %3d, Cent Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))

                if ((i+1) % save_period == 0):
                    torch.save(avg_model.state_dict(), 'Output/%s/%s/%d_com_sel.pt' %(data_obj.name, method_name, (i+1)))     
                    # torch.save(all_model.state_dict(), 'Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (i+1)))
                    # np.save('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, (i+1)), clnt_params_list)

                    np.save('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, (i+1)), trn_perf_sel[:i+1])
                    np.save('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, (i+1)), tst_perf_sel[:i+1])

                    # np.save('Output/%s/%s/%d_com_trn_perf_all.npy' %(data_obj.name, method_name, (i+1)), trn_perf_all[:i+1])
                    # np.save('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, (i+1)), tst_perf_all[:i+1])

                if (i+1) > save_period:
                    if os.path.exists('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, i+1-save_period)):
                        # Delete the previous saved arrays
                        os.remove('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, i+1-save_period))
                        os.remove('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, i+1-save_period))

                        # os.remove('Output/%s/%s/%d_com_trn_perf_all.npy' %(data_obj.name, method_name, i+1-save_period))
                        # os.remove('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, i+1-save_period))

                        # os.remove('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, i+1-save_period))

            if ((i+1) % save_period == 0):
                fed_mdls_sel[i//save_period] = avg_model
                # fed_mdls_all[i//save_period] = all_model

    return fed_mdls_sel, trn_perf_sel, tst_perf_sel  # fed_mdls_all, trn_perf_all, tst_perf_all

 
















