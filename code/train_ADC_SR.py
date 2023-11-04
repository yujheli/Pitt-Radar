import argparse
import os
from math import log10

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pytorch_ssim
from data_utils import TrainDatasetFromFolder_radar3D_adc, ValDatasetFromFolder_radar3D_adc, display_transform
from loss import GeneratorLoss, GeneratorLoss_L1
from model import Generator_radar3D, UNet_3D, Generator_radar3D_adc
import util.helper as helper

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=128, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=1, type=int,
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=128, type=int, help='train epoch number')
parser.add_argument('--batch_size', type=int, default=16, help='id to focus on')
parser.add_argument('--model_name', default=None, type=str, help='generator model epoch name')
parser.add_argument('--low_Azimuth', type=int, default=4, help='lr number of Azimuth')
parser.add_argument('--high_Azimuth', type=int, default=12, help='hr number of Azimuth')
parser.add_argument('--output', metavar='DIR', default='./out',
                    help='path to output folder. If not set, will be created in data folder')
parser.add_argument('--train_data', metavar='DIR', default='./train',
                    help='path to output folder. If not set, will be created in data folder') 
parser.add_argument('--val_data', metavar='DIR', default='./test',
                    help='path to output folder. If not set, will be created in data folder')  


# mode = 'extend'
# mode = 'lap-extend'
# mode = 'eval-ssr'
# mode = 'else'

# mode = 'extend'
# mode = 'extend2'
mode = 'normal'
# mode = 'extend3'

num_low_receiver = 4

num_angle_bins = 256
# num_angle_bins = 128

# debug = True
debug = False
abs_vis = True
tune_voc = False
beg_voc = 40

def RAD_map(range_plot):
    range_plot = np.fft.fft(range_plot, axis=0)
    range_doppler = np.fft.fft(range_plot, axis=2)
    range_doppler = np.fft.fftshift(range_doppler, axes=2)
    padding = ((0,0), (0,num_angle_bins-range_doppler.shape[1]), (0,0))
    range_azimuth = np.pad(range_doppler, padding, mode='constant')

    # import pdb
    # pdb.set_trace()
    range_azimuth = np.fft.fft(range_azimuth, axis=1)
    range_azimuth = np.fft.fftshift(range_azimuth, axes=1)
    out_img = np.rot90(range_azimuth,2,axes=(0,1))
    # out_img = range_azimuth
    return out_img

if __name__ == '__main__':
    opt = parser.parse_args()
    
    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    out_path = opt.output

    train_data_adc_dir = opt.train_data
    val_data_adc_dir = opt.val_data

    num_low_receiver = opt.low_Azimuth
    num_high_receiver = opt.high_Azimuth

    print("Inupt train dir",train_data_adc_dir)
    print("Inupt test dir",val_data_adc_dir)
    
    train_set = TrainDatasetFromFolder_radar3D_adc(train_data_adc_dir, num_low_receiver, None, None, crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR,\
        index_list=1, num_high_receiver=num_high_receiver)
    val_set = ValDatasetFromFolder_radar3D_adc(val_data_adc_dir, num_low_receiver, None, None, upscale_factor=UPSCALE_FACTOR, index_list=1,\
        num_high_receiver=num_high_receiver)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=opt.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)
    
    # netG = Generator_radar3D(UPSCALE_FACTOR)
    UPSCALE_FACTOR = opt.high_Azimuth/num_low_receiver
    netG =Generator_radar3D_adc(UPSCALE_FACTOR,input_dim=2)
    # netG = UNet_3D()

    if opt.model_name is not None:
        netG.load_state_dict(torch.load(opt.model_name))
        print("Pre-trained weights loaded")

    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    # netD = Discriminator_radar()
    # print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
    
    generator_criterion = GeneratorLoss()
    # generator_criterion = GeneratorLoss_L1()
    
    if torch.cuda.is_available():
        netG.cuda()
        # netD.cuda()
        generator_criterion.cuda()
    
    optimizerG = optim.Adam(netG.parameters())
    # optimizerD = optim.Adam(netD.parameters())
    
    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
    
    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
    
        netG.train()
        # netD.train()
        for adc, adc_data_low, data, target in train_bar:
            if debug:
                break

            # import pdb
            # pdb.set_trace()
            g_update_first = True
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size
    
            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################



            # real_img = Variable(target)
            # adc_data_low = Variable()
            # if torch.cuda.is_available():
            #     real_img = real_img.cuda()

            z = Variable(adc_data_low)
            y = Variable(adc)

            if torch.cuda.is_available():
                z = z.cuda()
                y = y.cuda()
            # fake_img = netG(z)
            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            netG.zero_grad()
            ## The two lines below are added to prevent runetime error in Google Colab ##
            # fake_img = netG(z)
            fake_adc = netG(z)
            # fake_out = netD(fake_img).mean()
            fake_out=None
            ##
            # import pdb
            # pdb.set_trace()
            g_loss = generator_criterion(fake_out, fake_adc, y)
            g_loss.backward()
            
            # fake_img = netG(z)
            # fake_out = netD(fake_img).mean()
            
            
            optimizerG.step()

            # loss for current batch before optimization 
            running_results['g_loss'] += g_loss.item() * batch_size
            # running_results['g2_loss'] += g2_loss.item() * batch_size
            # running_results['d_loss'] += d_loss.item() * batch_size
            # running_results['d_score'] += real_out.item() * batch_size
            # running_results['g_score'] += fake_out.item() * batch_size

            desp = '[%d/%d] Loss_G1: %.4f' % (epoch, NUM_EPOCHS,
                running_results['g_loss'] / running_results['batch_sizes'])
            train_bar.set_description((desp))
    
        netG.eval()
        
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
        with torch.no_grad():
            val_bar = tqdm(val_loader) # batch size 1
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []

            i = 0
            # for adc, adc_data_low, data, target in val_bar:
            for adc_hr, adc_lr, val_lr, val_hr in val_bar:
                batch_size = val_lr.size(0)

                #1 2 A D R --> RAD2
                valing_results['batch_sizes'] += batch_size
                # lr = val_lr
                # hr = val_hr
                if torch.cuda.is_available():
                    # lr = lr.cuda()
                    # hr = hr.cuda()
                    adc_lr = adc_lr.cuda()
                    adc_hr =  adc_hr.cuda()
                adc_sr = netG(adc_lr)

                # Doing something here for SSR
                # import pdb
                # pdb.set_trace()
                # adc_sr = netG(adc_sr[:,:,:4,:,:])
                if mode == 'extend':
                    adc_sr = netG(adc_sr)
                elif mode == 'extend2':
                    adc_sr = netG(adc_sr)
                    adc_sr = netG(adc_sr)
                elif mode == 'extend3':
                    adc_sr = netG(adc_sr)
                    adc_sr = netG(adc_sr)
                    adc_sr = netG(adc_sr)
                elif mode == 'lap-extend':
                    extend_dim = 4
                    adc_sr_left = netG(adc_sr[:,:,:extend_dim,:,:].clone())[:,:,:extend_dim,:,:]
                    adc_sr_right = netG(adc_sr[:,:,-extend_dim:,:,:].clone())[:,:,-extend_dim:,:,:]


                    adc_sr = torch.cat((adc_sr_left,adc_sr,adc_sr_right),2)
                    # import pdb
                    # pdb.set_trace()

                elif mode == 'eval-ssr':
                    adc_sr = netG(adc_hr)[:,:,:12,:,:]
                    # adc_sr = netG(adc_sr)

                else:
                    pass
                
                # visualize signal
                adc_sr = adc_sr.squeeze()*100 # 2 A R D
                adc_sr = adc_sr.permute(2,1,3,0) # R A D 2
                # import pdb
                # pdb.set_trace()
                # adc_sr = torch.view_as_complex(adc_sr).cpu().numpy()
                adc_sr = (adc_sr[...,0] + 1j * adc_sr[...,1]).cpu().numpy() # R A D
                rad_sr = RAD_map(adc_sr)
                #FFTs
                if tune_voc:
                    rad_sr = rad_sr[:,:,beg_voc:]

                sr_img_ra = helper.getLog(helper.getSumDim(helper.getMagnitude(rad_sr, power_order=1), \
                                            target_axis=-1), scalar=10, log_10=True)

                if abs_vis:
                    sr_img = helper.norm2Image_abs2(sr_img_ra)[..., :3]
                else:
                    sr_img = helper.norm2Image(sr_img_ra)[..., :3]

                adc_lr = adc_lr.squeeze()*100
                adc_lr = adc_lr.permute(2,1,3,0)
                # adc_lr = torch.view_as_complex(adc_lr).cpu().numpy()
                adc_lr = (adc_lr[...,0] + 1j * adc_lr[...,1]).cpu().numpy()
                rad_lr = RAD_map(adc_lr)
                lr_img_ra = helper.getLog(helper.getSumDim(helper.getMagnitude(rad_lr, power_order=1), \
                                            target_axis=-1), scalar=10, log_10=True)

                if abs_vis:                            
                    lr_img = helper.norm2Image_abs2(lr_img_ra)[..., :3]
                else:
                    lr_img = helper.norm2Image(lr_img_ra)[..., :3]
                
                adc_hr = adc_hr.squeeze()*100
                adc_hr = adc_hr.permute(2,1,3,0)
                # adc_lr = torch.view_as_complex(adc_lr).cpu().numpy()
                adc_hr = (adc_hr[...,0] + 1j * adc_hr[...,1]).cpu().numpy()
                rad_hr = RAD_map(adc_hr)
                hr_img_ra = helper.getLog(helper.getSumDim(helper.getMagnitude(rad_hr, power_order=1), \
                                            target_axis=-1), scalar=10, log_10=True)
                if abs_vis:                            
                    hr_img = helper.norm2Image_abs2(hr_img_ra)[..., :3]
                else:
                    hr_img = helper.norm2Image(hr_img_ra)[..., :3]
                

                # lr = lr.squeeze()*10
                # # lr = lr.squeeze()
                # lr = (torch.pow(10,lr)-1.).cpu().numpy().transpose(1,2,0)

                # if tune_voc:
                #     lr = lr[:,:,beg_voc:]

                # lr = helper.getLog(helper.getSumDim(lr, \
                #                             target_axis=-1), scalar=10, log_10=True)
                # lr_img = helper.norm2Image_abs(lr)[..., :3]
               
                # hr = hr.squeeze()*10
                # # hr = hr.squeeze()
                # hr = (torch.pow(10,hr)-1.).cpu().numpy().transpose(1,2,0)

                # if tune_voc:
                #     hr = hr[:,:,beg_voc:]

                # hr = helper.getLog(helper.getSumDim(hr, \
                #                             target_axis=-1), scalar=10, log_10=True)
                # hr_img = helper.norm2Image_abs(hr)[..., :3]


                # import pdb
                # pdb.set_trace()
                
                # batch_mse = ((sr - hr) ** 2).mean()
                # batch_mse = ((sr_img_ra - hr_img_ra)/255. ** 2).mean()
                batch_mse = ((sr_img_ra - hr_img_ra)** 2).mean()
                batch_mse = ((np.log2(abs(rad_sr)) - np.log2(abs(rad_hr)))** 2).mean()
                # print_batch_mse = ((RA_img - hr_img) ** 2).mean()
                valing_results['mse'] += batch_mse * batch_size
                valing_results['mse_eval'] = (valing_results['mse'] / valing_results['batch_sizes'])
                # batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                # valing_results['ssims'] += batch_ssim * batch_size
                hr_img_ra = np.log2(abs(rad_hr))
                valing_results['psnr'] = 10 * log10((hr_img_ra.max()**2) / (valing_results['mse'] / valing_results['batch_sizes']))
                # valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB' % (
                        valing_results['psnr']))
                # RD_img = helper.norm2Image(RD)[..., :3]

                # val_images.extend(
                #     [display_transform()(torch.mean(torch.exp2(val_hr_restore),1).data.cpu()), display_transform()(torch.mean(hr,1).data.cpu()),
                #      display_transform()(torch.mean(sr,1).data.cpu())])
                # val_images.extend(
                #     [display_transform()(torch.log2(torch.sum(torch.exp2(val_hr_restore),1).data.cpu())), display_transform()(torch.log2(torch.sum(torch.exp2(hr),1).data.cpu())),
                #      display_transform()(torch.log2(torch.sum(torch.exp2(sr),1).data.cpu()))])
                # val_images.extend(
                #     [display_transform()((torch.log2(torch.mean(torch.exp2(val_hr_restore),1).data.cpu())-10)/10), display_transform()((torch.log2(torch.mean(torch.exp2(hr),1).data.cpu())-10)/10),
                #      display_transform()((torch.log2(torch.mean(torch.exp2(sr),1).data.cpu())-10)/10)])
                # val_images.extend(
                #     [display_transform()((torch.log2(torch.mean(torch.exp2(val_hr_restore),1).data.cpu())-10)/10), display_transform()((torch.log2(torch.mean(torch.exp2(hr),1).data.cpu())-10)/10),
                #      display_transform()((torch.log2(torch.mean(torch.exp2(sr),1).data.cpu())-10)/10)])
                
                # range 0-1
                # import pdb
                # pdb.set_trace()
                val_images.extend(
                    [display_transform()(lr_img), display_transform()(hr_img),
                     display_transform()(sr_img)])
                
                i+=1
                # if i>=5:
                #     break
               

            # import pdb
            # pdb.set_trace()
            len_val = (len(val_images)//15)*15
            val_images = val_images[:len_val]
            val_images = torch.stack(val_images)
            val_images = torch.chunk(val_images, val_images.size(0) // 15)
            val_save_bar = tqdm(val_images, desc='[saving training results]')
            index = 1
            for image in val_save_bar:
                # import pdb
                # pdb.set_trace()
                image = utils.make_grid(image, nrow=3, padding=5)
                utils.save_image(image, out_path + 'epoch_%d_index_%d_%.4f.png' % (epoch, index, valing_results['mse_eval']), padding=5)
                index += 1
    
        # save model parameters
        torch.save(netG.state_dict(), out_path + 'netG_epoch_%d.pth' % (epoch))
        # torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        # torch.save(netD.state_dict(), 'epochs/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        # save loss\scores\psnr\ssim
        # results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        # results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        # results['ssim'].append(valing_results['ssim'])
    
        # if (epoch+1) % 10 == 0 and epoch != 0:
        #     out_path = 'statistics/'
        #     data_frame = pd.DataFrame(
        #         data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
        #               'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
        #         index=range(1, epoch + 1))
        #     data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_train_results.csv', index_label='Epoch')
