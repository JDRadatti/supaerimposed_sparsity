import sparse
import torch
from pathlib import Path
import vgg16_act


def create_sparse_vgg16(nw,mw,na,ma,setting_w,setting_a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = vgg16_act.vgg16_act_cifar10(N=1, M=1, batch_norm=True, classifier_hidden=512).to(device)
    ckpt = "./runs/vgg16_cifar10/vgg16_nosparse.pt"
    ckpt_path=Path(ckpt)
    start_epoch, best_acc = vgg16_act.load_checkpoint(ckpt_path, model, optimizer=None, scaler=None, strict=True)
    _ =sparse.prune_nm(model,nw,mw,setting_w)  
    ckpt = "./runs/vgg16_cifar10/vgg16_w{}_{}_{}.pt".format(setting_w,nw,mw)
    ckpt_path=Path(ckpt)
    vgg16_act.train_and_save(model,10,ckpt_path,lr=0.04)

    model = vgg16_act.vgg16_act_cifar10(N=na,M=ma,batch_norm=True,classifier_hidden=512,num_classes=10,conv_mask_dim=setting_a,linear_mask_dim=setting_a).to(device)
    _ =sparse.prune_nm(model,nw,mw,setting_w)  
    start_epoch, best_acc = vgg16_act.load_checkpoint(ckpt_path, model, optimizer=None, scaler=None, strict=True)
    ckpt = "./runs/vgg16_cifar10/vgg16_{}_{}_{}_act_{}_{}_{}.pt".format(setting_w,nw,mw,setting_a,na,ma)
    ckpt_path=Path(ckpt)
    vgg16_act.train_and_save(model,30,ckpt_path,lr=0.04)


def create_sparse_vgg16_short(nw,mw,na,ma,setting_w,setting_a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = "./runs/vgg16_cifar10/vgg16_w{}_{}_{}.pt".format(setting_w,nw,mw)
    ckpt_path=Path(ckpt)

    model = vgg16_act.vgg16_act_cifar10(N=na,M=ma,batch_norm=True,classifier_hidden=512,num_classes=10,conv_mask_dim=setting_a,linear_mask_dim=setting_a).to(device)
    _ =sparse.prune_nm(model,nw,mw,setting_w)  
    start_epoch, best_acc = vgg16_act.load_checkpoint(ckpt_path, model, optimizer=None, scaler=None, strict=True)
    ckpt = "./runs/vgg16_cifar10/vgg16_{}_{}_{}_act_{}_{}_{}.pt".format(setting_w,nw,mw,setting_a,na,ma)
    ckpt_path=Path(ckpt)
    vgg16_act.train_and_save(model,30,ckpt_path,lr=0.04)

def special_load(nw,mw,setting_w):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = "./runs/vgg16_w{}_{}_{}.pt".format(setting_w,nw,mw)
    ckpt_path=Path(ckpt)

    model = vgg16_act.vgg16_act_cifar10(N=1,M=1,batch_norm=True,classifier_hidden=512,num_classes=10,conv_mask_dim=0,linear_mask_dim=0).to(device)
    _ =sparse.prune_nm(model,nw,mw,setting_w)  
    start_epoch, best_acc = vgg16_act.load_checkpoint(ckpt_path, model, optimizer=None, scaler=None, strict=True)
    return model



def load_and_evaluate(nw,mw,na,ma,setting_w,setting_a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = "./runs/vgg16_cifar10/vgg16_{}_{}_{}_act_{}_{}_{}.pt".format(setting_w,nw,mw,setting_a,na,ma)
    ckpt_path=Path(ckpt)
    model = vgg16_act.vgg16_act_cifar10(N=na,M=ma,batch_norm=True,classifier_hidden=512,num_classes=10,conv_mask_dim=setting_a,linear_mask_dim=setting_a).to(device)
    _ =sparse.prune_nm(model,nw,mw,setting_w)  
    start_epoch, best_acc = vgg16_act.load_checkpoint(ckpt_path, model, optimizer=None, scaler=None, strict=True)
    print(vgg16_act.get_validation_accuracy(model))



def main():
    model=special_load(1,4,'row')
    print(model.modules)

  






    




if __name__ == "__main__":
     
     main()





