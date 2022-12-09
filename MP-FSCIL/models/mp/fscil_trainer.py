from .base import Trainer
import torch.nn as nn
from copy import deepcopy
from .helper import *
from utils import * 
from dataloader.data_utils import *


class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)

        self.pre_model = MYNET(self.args)
        self.pre_model = nn.DataParallel(self.pre_model, list(range(self.args.num_gpu)))
        self.pre_model = self.pre_model.cuda()

        self.meta_model = MYNET_Meta(self.args)
        self.meta_model = nn.DataParallel(self.meta_model, list(range(self.args.num_gpu)))
        self.meta_model = self.meta_model.cuda()


    def get_pre_optimizer(self):

        optimizer = torch.optim.SGD(self.pre_model.parameters(), self.args.pre_lr, momentum=0.9, nesterov=True,
                                    weight_decay=0.0005)
        if self.args.pre_schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.pre_step, gamma=0.1)
        elif self.args.pre_schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.pre_milestones,
                                                             gamma=0.1)

        return optimizer, scheduler

    def get_meta_optimizer(self):

        optimizer = torch.optim.SGD([{'params': self.meta_model.module.encoder.parameters(), 'lr': self.args.meta_lr}],
                                    momentum=0.9, nesterov=True, weight_decay=0.0005)
        # optimizer = torch.optim.SGD([{'params': self.model.module.encoder.parameters(), 'lr': self.args.lr_base}],
        #                             momentum=0.9, nesterov=True, weight_decay=self.args.decay)

        if self.args.meta_schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.meta_step, gamma=0.1)
        elif self.args.meta_schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.meta_milestones,
                                                             gamma=0.1)

        return optimizer, scheduler

    def get_dataloader(self, session, meta=False):
        if session == 0:
            if not meta:
                trainset, trainloader, testloader = get_pre_dataloader(self.args)
            else:
                trainset, trainloader, testloader = get_meta_dataloader(self.args)
        else:
            trainset, trainloader, testloader = get_new_dataloader(self.args, session)
        return trainset, trainloader, testloader
    
    def clean(self, args):
        self.trlog = {}
        self.trlog['train_loss'] = []
        self.trlog['val_loss'] = []
        self.trlog['test_loss'] = []
        self.trlog['train_acc'] = []
        self.trlog['val_acc'] = []
        self.trlog['test_acc'] = []
        self.trlog['max_acc_epoch'] = 0
        self.trlog['max_acc'] = [0.0] * args.sessions
    
    def train(self):
        args = self.args
        t_start_time = time.time()

        # init train statistics
        result_list = [args]

        for session in range(args.start_session, args.sessions):
            train_set, trainloader, testloader = self.get_dataloader(session, meta=False)
            if session == 0:  # load base class train img label
                # pre-training
                print('new classes for this session:\n', np.unique(train_set.targets))
                optimizer, scheduler = self.get_pre_optimizer()
                for epoch in range(args.pre_epochs+1):
                    start_time = time.time()
                    # train base sess
                    print("!!!!!!!!!!!!!!!!pre-train!!!!!!!!!!!!!!!!")
                    tl, ta = pre_train(self.pre_model, trainloader, optimizer, scheduler, epoch, args)
                    # test model with all seen class
                    tsl, tsa = test(self.pre_model, testloader, epoch, args, session, pri=True)
                    if epoch%10 == 0:
                        save_dir = os.path.join(args.save_path, str(epoch) + '.pth')
                        torch.save(dict(params=self.pre_model.state_dict()), save_dir)
                    # save better model
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        self.trlog['max_acc_epoch'] = epoch
                        save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                        torch.save(dict(params=self.pre_model.state_dict()), save_model_dir)
                        torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                        self.best_model_dict = deepcopy(self.pre_model.state_dict())
                        print('********A better model is found!!**********')
                        print('Saving model to :%s' % save_model_dir)
                    print('best epoch {}, best test acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                       self.trlog['max_acc'][session]))

                    self.trlog['train_loss'].append(tl)
                    self.trlog['train_acc'].append(ta)
                    self.trlog['test_loss'].append(tsl)
                    self.trlog['test_acc'].append(tsa)
                    lrc = scheduler.get_last_lr()[0]
                    result_list.append(
                        'epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                            epoch, lrc, tl, ta, tsl, tsa))
                    print('This epoch takes %d seconds' % (time.time() - start_time),
                          '\nstill need around %.2f mins to finish this session' % (
                                  (time.time() - start_time) * (args.pre_epochs - epoch) / 60))
                    scheduler.step()

                #meta-training
                print("!!!!!!!!!!!!!!!!meta-train!!!!!!!!!!!!!!!!")
                self.clean(args)
                train_set, trainloader, testloader = self.get_dataloader(session, meta=True)
                self.meta_model = update_param(self.meta_model, self.best_model_dict)
                self.meta_model.eval()
                optimizer, scheduler = self.get_meta_optimizer()
                for epoch in range(args.meta_episode+1):
                    start_time = time.time()
                    # train base sess
                    tl, ta = meta_train(self.meta_model, train_set, trainloader, optimizer, scheduler, epoch, args)
                    self.meta_model = replace_base_fc(train_set, testloader.dataset.transform, self.meta_model, args)
                    if epoch%10 == 0:
                        vl, va = self.validation()
                        if epoch % 10 == 0:
                            save_model_dir = os.path.join(args.save_path, str(epoch) + 'meta.pth')
                            torch.save(dict(params=self.meta_model.state_dict()), save_model_dir)

                        # save better model
                        if (va * 100) >= self.trlog['max_acc'][session]:
                            self.trlog['max_acc'][session] = float('%.3f' % (va * 100))
                            self.trlog['max_acc_epoch'] = epoch
                            save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                            torch.save(dict(params=self.meta_model.state_dict()), save_model_dir)
                            torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                            self.best_model_dict = deepcopy(self.meta_model.state_dict())
                            print('********A better model is found!!**********')
                            print('Saving model to :%s' % save_model_dir)
                        print('best epoch {}, best val acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                        self.trlog['max_acc'][session]))
                        self.trlog['val_loss'].append(vl)
                        self.trlog['val_acc'].append(va)
                        lrc = scheduler.get_last_lr()[0]
                        print('epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,val_loss:%.5f,val_acc:%.5f' % (
                            epoch, lrc, tl, ta, vl, va))
                        result_list.append(
                            'epoch:%03d,lr:%.5f,training_loss:%.5f,training_acc:%.5f,val_loss:%.5f,val_acc:%.5f' % (
                                epoch, lrc, tl, ta, vl, va))
                    
                    self.trlog['train_loss'].append(tl)
                    self.trlog['train_acc'].append(ta)
                    print('This epoch takes %d seconds' % (time.time() - start_time),
                          '\nstill need around %.2f mins to finish' % (
                                  (time.time() - start_time) * (args.meta_episode - epoch) / 60))
                    scheduler.step()
                
                self.meta_model.load_state_dict(self.best_model_dict)
                self.meta_model = replace_base_fc(train_set, testloader.dataset.transform, self.meta_model, args)
                best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                self.best_model_dict = deepcopy(self.meta_model.state_dict())
                torch.save(dict(params=self.meta_model.state_dict()), best_model_dir)

                
                tsl, tsa = test(self.meta_model, testloader, 0, args, session)
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                print('The test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))

                result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session]))

            else:  # incremental learning sessions
                print("training session: [%d]" % session)
                self.meta_model.eval()
                trainloader.dataset.transform = testloader.dataset.transform
                self.meta_model.module.is_trans = False
                self.meta_model.module.update_fc(trainloader, np.unique(train_set.targets), session, 0)
                train_set_0, _, _ = self.get_dataloader(0)
                self.meta_model = replace_base_fc(train_set_0, testloader.dataset.transform, self.meta_model, args)
                self.meta_model.module.is_trans = True
                self.meta_model.module.update_fc(trainloader, np.unique(train_set.targets), session, 1)
                # self.meta_model.module.is_trans = True
                tsl, tsa = test(self.meta_model, testloader,0 , args, session, pri=True)
                # save model
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                torch.save(dict(params=self.meta_model.state_dict()), save_model_dir)
                self.best_model_dict = deepcopy(self.meta_model.state_dict())
                print('Saving model to :%s' % save_model_dir)
                print('  test acc={:.3f}'.format(self.trlog['max_acc'][session]))
                result_list.append('Session {}, test Acc {:.3f}\n'.format(session, self.trlog['max_acc'][session]))

        result_list.append('Base Session Best Epoch {}\n'.format(self.trlog['max_acc_epoch']))
        result_list.append(self.trlog['max_acc'])
        print(self.trlog['max_acc'])
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Base Session Best epoch:', self.trlog['max_acc_epoch'])
        print('Total time used %.2f mins' % total_time)

    def set_save_path(self):
        # mode = self.args.base_mode + '-' + self.args.new_mode
        
        # mode = mode + '-' + 'data_init'
        mode = 'data_init'

        self.args.save_path = '%s/' % self.args.dataset
        # self.args.save_path = self.args.save_path + '%s/' % self.args.project

        self.args.save_path = self.args.save_path + '%s-start_%d/' % (mode, self.args.start_session)
        if self.args.pre_schedule == 'Milestone':
            mile_stone = str(self.args.pre_milestones).replace(" ", "").replace(',', '_')[1:-1]
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-MS_%s-Bs_%d' % (
                self.args.pre_epochs, self.args.pre_lr, mile_stone, self.args.pre_batch_size)
        elif self.args.pre_schedule == 'Step':
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-Step_%d-Bs_%d' % (
                self.args.pre_epochs, self.args.pre_lr, self.args.pre_step, self.args.pre_batch_size)
        # if 'cos' in mode:
        self.args.save_path = self.args.save_path + '-T_%.2f' % (self.args.temperature)

        # if 'ft' in self.args.new_mode:
        #     self.args.save_path = self.args.save_path + '-ftLR_%.3f-ftEpoch_%d' % (
        #         self.args.lr_new, self.args.epochs_new)
        
        self.args.save_path = self.args.save_path + 'num' + str(self.args.num)

        if self.args.debug:
            self.args.save_path = os.path.join('debug', self.args.save_path)

        self.args.save_path = os.path.join('checkpoint', self.args.save_path)
        ensure_path(self.args.save_path)
        return None
    
    def validation(self):
        with torch.no_grad():
            meta_model = self.meta_model
            for session in range(1, self.args.sessions):
                train_set, trainloader, testloader = self.get_dataloader(session)
                trainloader.dataset.transform = testloader.dataset.transform
                meta_model.eval()
                self.meta_model.module.is_trans = False
                self.meta_model.module.update_fc(trainloader, np.unique(train_set.targets), session, 0)
                train_set_0, _, _ = self.get_dataloader(0)
                self.meta_model = replace_base_fc(train_set_0, testloader.dataset.transform, self.meta_model, self.args)
                self.meta_model.module.is_trans = True
                self.meta_model.module.update_fc(trainloader, np.unique(train_set.targets), session, 1)
                vl, va = test(meta_model, testloader, 0, self.args, session, pri=False)
        return vl, va
    def test(self, model, testloader, args, session, epoch=0, pri=False):
        test_class = args.base_class + session * args.way
        model = model.eval()
        vl = Averager()
        va = Averager()
        with torch.no_grad():
            for i, batch in enumerate(testloader, 1):    
                data, test_label = [_.cuda() for _ in batch]
                model.module.mode = 'encoder'
                # model.module.is_trans = True
                query = model(data)
                query = query.unsqueeze(0).unsqueeze(0)
                proto = model.module.fc[1].weight[:test_class, :].detach()
                proto = proto.unsqueeze(0).unsqueeze(0)
                logits = model.module._forward(proto, query)
                loss = F.cross_entropy(logits, test_label)
                acc = count_acc(logits, test_label)
                vl.add(loss.item())
                va.add(acc)
            vl = vl.item()
            va = va.item()
        if pri == True:
            print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))
        return vl, va