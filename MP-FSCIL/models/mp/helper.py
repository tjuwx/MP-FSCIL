# import new Network name here and add in model_class args
from sklearn.metrics import confusion_matrix
from .Network import MYNET 
from .Network import MYNET_Meta 
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
from dataloader.data_utils import *
from copy import deepcopy
import torch.nn as nn

def linear_combination(x, y, epsilon):  
    return epsilon*x + (1-epsilon)*y

def reduce_loss(loss, reduction='mean'): 
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss 

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon:float=0.1, reduction='mean'): 
        super().__init__() 
        self.epsilon = epsilon
        self.reduction = reduction 
 
    def forward(self, preds, target): 
        n = preds.size()[-1]
        log_preds = torch.log(F.softmax(preds, dim=-1)/16)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        # print(log_preds)
        # exit()
        return linear_combination(loss/n, nll, self.epsilon)

def update_param(model, pretrained_dict):
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model

def pre_train(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    ta = Averager()
    model = model.train()
    # standard classification for pretrain
    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]
        #RS loss
        model.module.mode = 'alpha'
        feature_maps = model(data)
        loss_rs = loss_maps(feature_maps, args.num)

        #CE loss
        model.module.mode = 'encoder'
        logits = model(data)
        logits = logits[:, :args.base_class]
        loss_cross = F.cross_entropy(logits, train_label)
        #total loss
        total_loss = loss_cross + loss_rs * 0.6
        acc = count_acc(logits, train_label)
        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    tl = tl.item()
    ta = ta.item()
    
    return tl, ta

def meta_train(model, train_set, trainloader, optimizer, scheduler, epoch, args):
        tl = Averager()
        ta = Averager()
        tqdm_gen = tqdm(trainloader)
        label = torch.arange(args.meta_way).repeat(args.meta_query)
        label = label.type(torch.cuda.LongTensor)
        for i, batch in enumerate(tqdm_gen, 1):
            data_img, true_label = [_.cuda() for _ in batch]
            k = args.meta_way * args.meta_shot
            model.module.mode = 'alpha'
            data = model(data_img)
            #RP loss
            data_tmp = model.module.weights(data)
            data_tmp = F.adaptive_avg_pool2d(data_tmp, 1).squeeze(-1).squeeze(-1)
            proto_tmp, query_tmp = data_tmp[:k], data_tmp[k:]
            proto_tmp = proto_tmp.view(1, args.meta_shot* args.meta_way, proto_tmp.shape[-1])
            query_tmp = query_tmp.view(args.meta_query, args.meta_way, query_tmp.shape[-1])
            proto_tmp = proto_tmp.unsqueeze(0)
            query_tmp = query_tmp.unsqueeze(0)
            logits = model.module._forward(proto_tmp, query_tmp)
            loss_fuction = LabelSmoothingCrossEntropy()
            rp_loss = loss_fuction.forward(logits, label)

            #TA loss
            data = F.adaptive_avg_pool2d(data, 1).squeeze(-1).squeeze(-1)
            proto, query = data[:k], data[k:]
            proto = proto.view(args.meta_shot, args.meta_way, proto.shape[-1])
            query = query.view(args.meta_query, args.meta_way, query.shape[-1])
            proto = proto.mean(0).unsqueeze(0)

            num = int(proto.size(1)/2)
            proto_base_in = proto[:,:num,:]
            proto = proto.reshape(proto.size(1), proto.size(0), -1)
            proto_base_ex = F.cosine_similarity(proto, proto_base_in, dim=2).unsqueeze(0)
            query = query.reshape(query.size(1)*query.size(0), 1, -1)
            query_ex = F.cosine_similarity(query, proto_base_in, dim=2)
            query_ex = query_ex.reshape(query_tmp.size(1), query_tmp.size(2), -1)
            proto_base_ex = proto_base_ex.unsqueeze(0)
            query_ex = query_ex.unsqueeze(0)
            logits = model.module._forward(proto_base_ex, query_ex) * 10
            ta_loss = F.cross_entropy(logits, label)
            total_loss =  ta_loss + rp_loss
            acc = count_acc(logits, label)
            lrc = scheduler.get_last_lr()[0]
            tqdm_gen.set_description(
                'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
            tl.add(total_loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        tl = tl.item()
        ta = ta.item()
        return tl, ta

def replace_base_fc(trainset, transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    
    embedding_list = []
    label_list = []
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.module.mode = 'encoder'
            model.module.is_trans = False
            embedding = model(data)
            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())          
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)
    proto_list = []
    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)
    proto_list = torch.stack(proto_list, dim=0)
    model.module.fc[0].weight.data[:args.base_class] = proto_list
    embedding_list = []
    label_list = []
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.module.mode = 'encoder'
            model.module.is_trans = True
            embedding = model(data)
            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())          
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)
    proto_list = []
    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)
    proto_list = torch.stack(proto_list, dim=0)
    model.module.fc[1].weight.data[:args.base_class] = proto_list

    return model

def test(model, testloader, epoch, args, session, pri=True):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()

    with torch.no_grad():
        if pri == True:
            tqdm_gen = tqdm(testloader)
        else:
            tqdm_gen = testloader
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            model.module.mode = 'encoder'
            logits = model(data)
            logits = logits[:, :test_class]
            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)
            vl.add(loss.item())
            va.add(acc)
        vl = vl.item()
        va = va.item()
    if pri == True:
        print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))
    
    return vl, va


# def get_dataloader(session, args):
#     if session == 0:
#         trainset, trainloader, testloader = get_meta_dataloader(args)
#     else:
#         trainset, trainloader, testloader = get_new_dataloader(args, session)
#     return trainset, trainloader, testloader


def loss_maps(feature_maps, number):
    feature_maps = F.relu(feature_maps)
    feature_maps_temp = feature_maps.view(feature_maps.size(0), feature_maps.size(1), -1)
    feature_maps_max, _ = torch.max(feature_maps_temp, dim=2, keepdim=True)
    feature_maps_norm = (feature_maps_temp)/(feature_maps_max+1e-9)
    sorted_tensor = torch.sort(feature_maps_norm, dim=2, descending=False)[0]
    head = torch.sum(sorted_tensor[:,:,-number:],dim=2)
    tail = torch.sum(sorted_tensor[:,:,:-number],dim=2)
    deno = (feature_maps_norm.size(0)*feature_maps_norm.size(1))
    loss = (torch.sum(F.relu(head-tail)))/deno
    return loss