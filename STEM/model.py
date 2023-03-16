import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from torch.nn import functional as F
import torch.optim.lr_scheduler as lr_scheduler
import os
import numpy as np
import time
from .utils import *
import scipy

class STData(Dataset):
    def __init__(self,data,coord):
        self.data = data
        self.coord = coord

    def __getitem__(self, index):
        return self.data[index], self.coord[index]
    
    def __len__(self):
        return self.data.shape[0]

class SCData(Dataset):
    def __init__(self,data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return self.data.shape[0]


class FeatureNet(nn.Module):
    def __init__(self, n_genes, n_embedding=[512,256,128],dp=0):
        super(FeatureNet, self).__init__()

        self.fc1 = nn.Linear(n_genes, n_embedding[0])
        self.bn1 = nn.LayerNorm(n_embedding[0])
        self.fc2 = nn.Linear(n_embedding[0], n_embedding[1])
        self.bn2 = nn.LayerNorm(n_embedding[1])
        self.fc3 = nn.Linear(n_embedding[1], n_embedding[2])
        
        self.dp = nn.Dropout(dp)
        
    def forward(self, x,isdp = False):
        if isdp:
            x = self.dp(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

class BaseModel(nn.Module):
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.device = opt.device

        self.train_log = opt.outf + '/train.log'
        self.model_path = opt.outf + '/model.pth'
        if not os.path.exists(opt.outf):
            os.mkdir(opt.outf)
        self.best_acc_tgt = 0
        
        with open(self.train_log, 'a') as f:
            localtime = time.asctime( time.localtime(time.time()) )
            f.write(localtime+str((opt.device,opt.outf,opt.n_genes,opt.no_bn,opt.lr,opt.sigma,opt.alpha,opt.verbose,opt.mmdbatch,opt.dp))+'\n')


    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def save(self):
        torch.save(self.state_dict(), self.model_path)

    def load(self):
        print('===> Loading model from {}'.format(self.model_path))
        try:
            self.load_state_dict(torch.load(self.model_path))
            print('<=== Success!')
        except:
            print('<==== Failed!')
            
class SOmodel(BaseModel):
    def __init__(self, opt):
        super(SOmodel, self).__init__(opt)
        self.netE = FeatureNet(opt.n_genes,dp=opt.dp)
        self.optimizer = torch.optim.AdamW(self.netE.parameters(), lr=opt.lr)
        self.lr_scheduler = lr_scheduler.StepLR(optimizer=self.optimizer,step_size=200, gamma=0.5)
        self.loss_names = ['E','E_pred','E_circle','E_mmd']
        self.mmd_fn = MMDLoss()
        self.sigma = opt.sigma
        self.alpha = opt.alpha
        self.verbose = opt.verbose
        self.mmdbatch = opt.mmdbatch
        
    def train_onestep(self,stdata,scdata,coord,ratio):
        if self.sigma == 0:
            self.nettrue = torch.eye(coord.shape[0])
        else:
            self.nettrue = torch.tensor(scipy.spatial.distance.cdist(coord,coord)).to(torch.float32)
            sigma = self.sigma
            self.nettrue = torch.exp(-self.nettrue**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)
            self.nettrue = F.normalize(self.nettrue,p=1,dim=1)
        self.nettrue = self.nettrue.to(self.device)
        stdata = stdata.to(self.device)
        scdata = scdata.to(self.device)
                
        self.e_seq_st = self.netE(stdata,True)
        self.e_seq_sc = self.netE(scdata,False)
        
        self.optimizer.zero_grad()
        
        self.netpred = self.e_seq_st.mm(self.e_seq_st.t())
        self.st2sc = F.softmax(self.e_seq_st.mm(self.e_seq_sc.t()),dim=1)
        self.sc2st = F.softmax(self.e_seq_sc.mm(self.e_seq_st.t()),dim=1)
        
        self.st2st = torch.log(self.st2sc.mm(self.sc2st)+1e-7)
        
        self.loss_E_pred = F.cross_entropy(self.netpred, self.nettrue,reduction='mean')
        
        self.loss_E_circle = F.kl_div(self.st2st,self.nettrue,reduction='none').sum(1).mean()
        
        ranidx = np.random.randint(0,self.e_seq_sc.shape[0],self.mmdbatch)
        
        self.loss_E_mmd = self.mmd_fn(self.e_seq_st,self.e_seq_sc[ranidx])
        
        self.loss_E = self.loss_E_pred + self.alpha*self.loss_E_mmd + ratio*self.loss_E_circle
        self.loss_E.backward()
        self.optimizer.step()
        
    def togpu(self):
        self.netE.to(self.device)

    def modeleval(self):
        self.netE.eval()
        
    def train(self,epoch,scdataloader,stdata,coord):
        with open(self.train_log, 'a') as f:
            localtime = time.asctime( time.localtime(time.time()) )
            f.write(localtime+'\n')
        
        loss_curve = {
            loss: []
            for loss in self.loss_names
        }
        for i in range(epoch):
            self.netE.train()
            for batch_idx, (scdata) in enumerate(scdataloader):
                scdata = scdata.to(torch.float32)
                self.train_onestep(stdata,scdata,coord,max((i-50)/(epoch-50),0))   
                for loss in self.loss_names:
                    loss_curve[loss].append(getattr(self, 'loss_' + loss).item())
                    
            self.lr_scheduler.step()
            loss_msg = '[Train][{}] Loss:'.format(i+1)
            for loss in self.loss_names:
                loss_msg += ' {} {:.3f}'.format(loss, loss_curve[loss][-1])
            if (i + 1) % 1 == 0:
                print(loss_msg)
                print(self.lr_scheduler.get_last_lr())
            with open(self.train_log, 'a') as f:
                f.write(loss_msg + "\n")
        return loss_curve
    
    
    def train_spatialbatch(self,epoch,scdata,stdataloader):
        with open(self.train_log, 'a') as f:
            localtime = time.asctime( time.localtime(time.time()) )
            f.write(localtime+'\n')
        
        loss_curve = {
            loss: []
            for loss in self.loss_names
        }
        for i in range(epoch):
            self.netE.train()
            self.netD.train()
            for batch_idx, (stdata,coord) in enumerate(stdataloader):
                stdata = stdata.to(torch.float32)
                self.train_onestep(stdata,scdata,coord,max((i-50)/(epoch-50),0))   
                for loss in self.loss_names:
                    loss_curve[loss].append(getattr(self, 'loss_' + loss).item())
                    
            for lr_scheduler in self.lr_schedulers:
                lr_scheduler.step()
            loss_msg = '[Train][{}] Loss:'.format(i+1)
            for loss in self.loss_names:
                loss_msg += ' {} {:.3f}'.format(loss, loss_curve[loss][-1])
            if (i + 1) % 1 == 0:
                print(loss_msg)
                print(self.lr_scheduler_D.get_last_lr(),self.lr_scheduler_G.get_last_lr())
            with open(self.train_log, 'a') as f:
                f.write(loss_msg + "\n")
        return loss_curve
    
    def train_wholedata(self,epoch,scdata,stdata,coord):
        with open(self.train_log, 'a') as f:
            localtime = time.asctime( time.localtime(time.time()) )
            f.write(localtime+'\n')
        
        loss_curve = {
            loss: []
            for loss in self.loss_names
        }
        for i in range(epoch):
            self.netE.train()
            scdata = scdata.to(torch.float32)
            if isinstance(stdata,list):
                shuffle_list = np.random.randint(0,len(stdata),len(stdata))
                for idx in shuffle_list:
                    self.train_onestep(stdata[idx],scdata,coord[idx],max((i-50)/(epoch-50),0))
            else:
                self.train_onestep(stdata,scdata,coord,max((i-50)/(epoch-50),0)) 
            
            for loss in self.loss_names:
                loss_curve[loss].append(getattr(self, 'loss_' + loss).item())
                    
            self.lr_scheduler.step()
            loss_msg = '[Train][{}] Loss:'.format(i)
            for loss in self.loss_names:
                loss_msg += ' {} {:.3f}'.format(loss, loss_curve[loss][-1])
            if (i + 1) % 1 == 0:
                if self.verbose:
                    print(loss_msg)
                    print(self.lr_scheduler.get_last_lr())
            with open(self.train_log, 'a') as f:
                f.write(loss_msg + "\n")
        return loss_curve