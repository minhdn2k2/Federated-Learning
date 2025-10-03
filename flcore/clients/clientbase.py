import copy
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

from utils.utils_dataset import MultiDataset


class Client(object):
    def __init__(self, args, id, train_x, train_y):
        self.model = None
        self.dataset_name = args.dataset_name
        self.device = args.device
        self.id = id

        self.num_classes = args.num_classes
        self.train_x = train_x
        self.train_y = train_y
        self.num_samples = len(self.train_x)

        self.batch_size = args.batch_size
        self.local_learning_rate = args.local_learning_rate
        self.lr_decay = args.lr_decay
        self.weight_decay = args.weight_decay
        self.local_epochs = args.local_epochs

        self.loss = nn.CrossEntropyLoss(reduction='sum')
        self.momentum = args.momentum


    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        dataset = MultiDataset(self.train_x, self.train_y,
                               train=True, dataset_name=self.dataset_name)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)


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


    

    


        
    


        


