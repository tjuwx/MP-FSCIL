import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cub_encoder import *
from models.mini_encoder import *
from models.cifar_encoder import *

class MYNET(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.weights = nn.Conv2d(512, 512, 1)
        self.is_trans = False
        if self.args.dataset in ['cifar100']:
            self.encoder = resnet_cifar()
            self.num_features = 512
        if self.args.dataset in ['mini_imagenet']:
            self.encoder = resnet_mini() 
        if self.args.dataset == 'cub200':
            self.encoder = resnet_cub(True, args)  # pretrained=True follow TOPIC, models for cub is imagenet pre-trained. https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
        self.num_features = 512 
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)
        self.fc = nn.Sequential(
            nn.Linear(self.num_features, 200, bias=False),
            nn.Linear(200, self.args.num_classes, bias=False)
        )


    def encode(self, x):
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        if self.is_trans:
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc[0].weight, p=2, dim=-1))
            x = self.args.temperature * x
        return x

    def forward(self, input):
        if self.mode == 'alpha':
            feature_maps = self.encoder(input)
            return feature_maps
        elif self.mode == 'encoder':
            input = self.encode(input)
            return input
        else:
            raise ValueError('Unknown mode')

    def update_fc(self,dataloader,class_list,session, i):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            data=self.encode(data).detach()
        new_fc = self.update_fc_avg(data, label, class_list, i)


    def update_fc_avg(self,data,label,class_list,i):
        new_fc=[]
        for class_index in class_list:
            data_index=(label==class_index).nonzero().squeeze(-1)
            embedding=data[data_index]
            proto=embedding.mean(0)
            new_fc.append(proto)
            self.fc[i].weight.data[class_index]=proto
        new_fc=torch.stack(new_fc,dim=0)
        return new_fc

class MYNET_Meta(MYNET):
    def __init__(self, args, mode=None):
        super().__init__(args)
        
    def forward(self, input):
        if self.mode == 'encoder':
            input = self.encode(input)
            return input
        elif self.mode == 'alpha':
            feature_maps = self.encoder(input)
            return feature_maps
        else:
            support_idx, query_idx = input
            logits = self._forward(support_idx, query_idx)
            return logits
    
    def encode(self, x):
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        if self.is_trans:
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc[0].weight, p=2, dim=-1))
            x = self.args.temperature * x
        return x

    def _forward(self, support, query):
        emb_dim = support.size(-1)
        # get mean of the support
        proto = support.mean(dim=1)
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        num_query = query.shape[1]*query.shape[2]#num of query*way
        query = query.view(-1, emb_dim).unsqueeze(1)
        proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
        proto = proto.view(num_batch*num_query, num_proto, emb_dim)
        logits=F.cosine_similarity(query, proto, dim=-1)
        logits=logits*self.args.temperature
        return logits
