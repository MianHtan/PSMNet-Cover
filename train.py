import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from pathlib import Path

from utils.stereo_datasets import fetch_dataset
from PSMNet.PSMNet import PSMNet

class Tripleloss(nn.Module):
    def __init__(self):
        super(Tripleloss, self).__init__()
    def forward(self, gt, pred1, pred2, pred3):
        loss = 0.5*F.smooth_l1_loss(pred1, gt, reduction='mean') + 0.7*F.smooth_l1_loss(pred2, gt, reduction='mean') + F.smooth_l1_loss(pred3, gt, reduction='mean')
        return loss

def train(net, dataset_name, batch_size, root, min_disp, max_disp, iters, init_lr, resize, device, save_frequency=None, require_validation=False, pretrain = None):
    print("Train on:", device)
    Path("training_checkpoints").mkdir(exist_ok=True, parents=True)

    # tensorboard log file
    writer = SummaryWriter(log_dir='logs')

    # define model
    net.to(device)
    if pretrain is not None:
        net.load_state_dict(torch.load(pretrain), strict=True)
        print("Finish loading pretrain model!")
    else:
        net._init_params()
        print("Model parameters has been random initialize!")
    net.train()

    # fetch traning data
    train_loader = fetch_dataset(dataset_name = dataset_name, root = root,
                                batch_size = batch_size, resize = resize, 
                                min_disp = min_disp, max_disp = max_disp, mode = 'training')
    
    steps_per_iter = train_loader.__len__()
    num_steps = steps_per_iter * iters    
    
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in net.parameters()])))
    # initialize the optimizer and lr scheduler
    optimizer = torch.optim.AdamW(net.parameters(), lr=init_lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, init_lr, num_steps + 100,
                                              pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')
    
    criterion = Tripleloss().to(device)

    # start traning
    should_keep_training = True
    total_steps = 0
    while should_keep_training:
        print('--- start new epoch ---')
        for i_batch, data_blob in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            image1, image2, disp_gt, valid = [x.to(device) for x in data_blob]
            valid = valid.detach_()

            net.training
            pred1, pred2, pred3 = net(image1, image2, min_disp, max_disp)
            assert net.training

            loss = criterion(pred1[valid], pred2[valid], pred3[valid], disp_gt[valid])
            loss.backward()
            optimizer.step()
            scheduler.step()
            # code of validation
            if total_steps % save_frequency == (save_frequency - 1):
                # save checkpoints
                save_path = Path('training_checkpoints/%dsteps_PSMNet_%s.pth' % (total_steps + 1, dataset_name))
                torch.save(net.state_dict(), save_path)

                # load validation data 
                if require_validation:
                    print("--- start validation ---")
                    test_loader = fetch_dataset(dataset_name = dataset_name, root = root,
                                    batch_size = batch_size, resize = resize, 
                                    min_disp = min_disp, max_disp = max_disp, mode = 'testing')
                    
                    val_loss_train = 0
                    val_loss_eval = 0
                    with torch.no_grad():
                        for i_batch, data_blob in enumerate(tqdm(test_loader)):
                            image1, image2, disp_gt, valid = [x.to(device) for x in data_blob]

                            net.eval()
                            pred = net(image1, image2, min_disp, max_disp)
                            val_loss_eval += F.smooth_l1_loss(pred[valid], disp_gt[valid], reduction='mean')

                            net.train()
                            pred1, pred2, pred3 = net(image1, image2, min_disp, max_disp)
                            val_loss_train += F.smooth_l1_loss(pred3[valid], disp_gt[valid], reduction='mean')
                        val_loss_eval = val_loss_eval / test_loader.__len__()
                        val_loss_train = val_loss_train / test_loader.__len__()
                    writer.add_scalars(main_tag="loss/vaildation loss", tag_scalar_dict = {'train()': val_loss_train}, global_step=total_steps+1)
                    writer.add_scalars(main_tag="loss/vaildation loss", tag_scalar_dict = {'eval()':val_loss_eval}, global_step=total_steps+1)

                net.train()
            
            # write loss and lr to log
            writer.add_scalar(tag="loss/training loss", scalar_value=loss, global_step=total_steps+1)
            writer.add_scalar(tag="lr/lr", scalar_value=scheduler.get_last_lr()[0], global_step=total_steps+1)
            total_steps += 1

            if total_steps > num_steps:
                should_keep_training = False
                break

        if len(train_loader) >= 1000:
            cur_iter = int(total_steps/steps_per_iter)
            save_path = Path('training_checkpoints/%d_epoch_PSMNet_%s.pth' % (cur_iter, dataset_name))
            torch.save(net.state_dict(), save_path)

    print("FINISHED TRAINING")

    final_outpath = f'training_checkpoints/PSMNet_{dataset_name}.pth'
    torch.save(net.state_dict(), final_outpath)
    print("model has been save to path: ", final_outpath)

if __name__ == '__main__':
    

    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

    net = PSMNet()

    # training set keywords: 'DFC2019', 'WHUStereo', 'all'
    # '/home/lab1/datasets/DFC2019_track2_grayscale_8bit'
    # '/home/lab1/datasets/whu_stereo_8bit/with_ground_truth'
    train(net=net, dataset_name='WHUStereo', root = '/home/lab1/datasets/whu_stereo_8bit/with_ground_truth', 
          batch_size=1, min_disp=-64, max_disp=64, iters=10, init_lr=0.0002,
          resize = [1024,1024], save_frequency = 500, require_validation=True, 
          device=device, pretrain='checkpoints/WHUStereo/PSMNet_WHUStereo.pth')