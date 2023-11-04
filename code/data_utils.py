from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import numpy as np
import torch
from glob import glob
import util.helper as helper

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def is_numpy_file(filename):
    return any(filename.endswith(extension) for extension in ['.npy'])

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        # RandomCrop(crop_size),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])

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

class TrainDatasetFromFolder(Dataset):
    def __init__(self, hr_data_dir, lr_data_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.hr_filenames = [join(hr_data_dir, x) for x in sorted(listdir(hr_data_dir)) if is_image_file(x)]
        self.lr_filenames = [join(lr_data_dir, x) for x in sorted(listdir(lr_data_dir)) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        # self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.hr_filenames[index]).convert('RGB'))
        lr_image = self.hr_transform(Image.open(self.lr_filenames[index]).convert('RGB'))
        # lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.hr_filenames)

def complexTo2Channels(target_array):
    """ transfer complex a + bi to [a, b]"""
    # assert target_array.dtype == np.complex64
    ### NOTE: transfer complex to (magnitude) ###
    output_array = getMagnitude(target_array)
    output_array = getLog(output_array)
    return output_array

def getMagnitude(target_array, power_order=1):
    """ get magnitude out of complex number """
    target_array = np.abs(target_array)
    target_array = pow(target_array, power_order)
    return target_array 

def getLog(target_array, scalar=1., log_10=True):
    """ get Log values """
    if log_10:
        return scalar * np.log10(target_array + 1.)
    else:
        return target_array


class TrainDatasetFromFolder_radar3D(Dataset):
    def __init__(self, hr_data_dir, lr_data_dir, crop_size, upscale_factor, index_list=None):
        super(TrainDatasetFromFolder_radar3D, self).__init__()
        if index_list != None:
            self.hr_filenames = [x for x in sorted(glob(hr_data_dir)) if is_numpy_file(x)]
            self.lr_filenames = [x for x in sorted(glob(lr_data_dir)) if is_numpy_file(x)]
            print(len(self.hr_filenames))
            print(len(self.lr_filenames))
            # import pdb
            # pdb.set_trace()
        else:
            self.hr_filenames = [join(hr_data_dir, x) for x in sorted(listdir(hr_data_dir)) if is_numpy_file(x)]
            self.lr_filenames = [join(lr_data_dir, x) for x in sorted(listdir(lr_data_dir)) if is_numpy_file(x)]


        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        # self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):

        # import pdb
        # pdb.set_trace()



        # hr_image = torch.from_numpy(np.log2(np.load(self.hr_filenames[index]).transpose(2,0,1))).type(torch.FloatTensor)
        # lr_image = torch.from_numpy(np.log2(np.load(self.lr_filenames[index]).transpose(2,0,1))).type(torch.FloatTensor)
        
        # hr_image = hr_image.unsqueeze(0)/13-1
        # lr_image = lr_image.unsqueeze(0)/13-1

        hr_data = complexTo2Channels(np.load(self.hr_filenames[index]))
        hr_image = torch.from_numpy(hr_data.transpose(2,0,1)).type(torch.FloatTensor)/10

        lr_data = complexTo2Channels(np.load(self.lr_filenames[index]))
        lr_image = torch.from_numpy(lr_data.transpose(2,0,1)).type(torch.FloatTensor)/10


        return lr_image[None], hr_image[None]

    def __len__(self):
        return len(self.hr_filenames)

class TrainDatasetFromFolder_radar2D(Dataset):
    def __init__(self, hr_data_dir, lr_data_dir, crop_size, upscale_factor, index_list=None):
        super().__init__()
        if index_list != None:
            self.hr_filenames = [x for x in sorted(glob(hr_data_dir)) if is_numpy_file(x)]
            self.lr_filenames = [x for x in sorted(glob(lr_data_dir)) if is_numpy_file(x)]
            print(len(self.hr_filenames))
            print(len(self.lr_filenames))
            # import pdb
            # pdb.set_trace()
        else:
            self.hr_filenames = [join(hr_data_dir, x) for x in sorted(listdir(hr_data_dir)) if is_numpy_file(x)]
            self.lr_filenames = [join(lr_data_dir, x) for x in sorted(listdir(lr_data_dir)) if is_numpy_file(x)]


        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        # self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        # hr_data = complexTo2Channels(np.load(self.hr_filenames[index]))
        # hr_image = torch.from_numpy(hr_data.transpose(2,0,1)).type(torch.FloatTensor)/10

        # import pdb
        # pdb.set_trace()
        hr_data = helper.getLog(helper.getSumDim(helper.getMagnitude(np.load(self.hr_filenames[index])\
            , power_order=1), target_axis=-1), scalar=10, log_10=True)
        # print(hr_data.shape)
        hr_image = torch.from_numpy(hr_data).type(torch.FloatTensor) / 10

        lr_data = helper.getLog(helper.getSumDim(helper.getMagnitude(np.load(self.lr_filenames[index])\
            , power_order=1), target_axis=-1), scalar=10, log_10=True)
        lr_image = torch.from_numpy(lr_data).type(torch.FloatTensor) / 10

        return lr_image[None], hr_image[None]

    def __len__(self):
        return len(self.hr_filenames)

class ValDatasetFromFolder_radar2D(Dataset):
    def __init__(self,  hr_data_dir, lr_data_dir, upscale_factor, index_list=None):
        super().__init__()
        self.upscale_factor = upscale_factor
        # self.image_filenames = [join(hr_data_dir, x) for x in listdir(hr_data_dir) if is_image_file(x)]
        if index_list != None:
            self.hr_filenames = [x for x in sorted(glob(hr_data_dir)) if is_numpy_file(x)]
            self.lr_filenames = [x for x in sorted(glob(lr_data_dir)) if is_numpy_file(x)]
            
        else:
            self.hr_filenames = [join(hr_data_dir, x) for x in sorted(listdir(hr_data_dir)) if is_numpy_file(x)]
            self.lr_filenames = [join(lr_data_dir, x) for x in sorted(listdir(lr_data_dir)) if is_numpy_file(x)]

        # import pdb
        # pdb.set_trace()
        # self.hr_filenames = [join(hr_data_dir, x) for x in sorted(listdir(hr_data_dir)) if is_numpy_file(x)]
        # self.lr_filenames = [join(lr_data_dir, x) for x in sorted(listdir(lr_data_dir)) if is_numpy_file(x)]

    def __getitem__(self, index):

        # hr_data = complexTo2Channels(np.load(self.hr_filenames[index]))
        # hr_image = torch.from_numpy(hr_data.transpose(2,0,1)).type(torch.FloatTensor)/10

        # lr_data = complexTo2Channels(np.load(self.lr_filenames[index]))
        # lr_image = torch.from_numpy(lr_data.transpose(2,0,1)).type(torch.FloatTensor)/10

        hr_data = helper.getLog(helper.getSumDim(helper.getMagnitude(np.load(self.hr_filenames[index])\
            , power_order=1), target_axis=-1), scalar=10, log_10=True)
        # print(hr_data.shape)
        hr_image = torch.from_numpy(hr_data).type(torch.FloatTensor) / 10

        lr_data = helper.getLog(helper.getSumDim(helper.getMagnitude(np.load(self.lr_filenames[index])\
            , power_order=1), target_axis=-1), scalar=10, log_10=True)
        lr_image = torch.from_numpy(lr_data).type(torch.FloatTensor) / 10


        return (lr_image[None]).type(torch.FloatTensor), (lr_image[None]).type(torch.FloatTensor), (hr_image[None]).type(torch.FloatTensor)

    def __len__(self):
        return len(self.hr_filenames)

class TrainDatasetFromFolder_radar3D_adc(Dataset):
    def __init__(self, adc_data_dir, num_low_receiver, hr_data_dir, lr_data_dir, crop_size, upscale_factor, index_list=None, num_high_receiver=12):
        super().__init__()
        if index_list != None: #Use this function for now
            self.adc_filenames = [x for x in sorted(glob(adc_data_dir+'*')) if is_numpy_file(x)]
            # self.hr_filenames = [x for x in sorted(glob(hr_data_dir)) if is_numpy_file(x)]
            # self.lr_filenames = [x for x in sorted(glob(lr_data_dir)) if is_numpy_file(x)]
            print(len(self.adc_filenames))
            # print(len(self.lr_filenames))
            # import pdb
            # pdb.set_trace()
        else:
            raise NotImplementedError
            self.hr_filenames = [join(hr_data_dir, x) for x in sorted(listdir(hr_data_dir)) if is_numpy_file(x)]
            self.lr_filenames = [join(lr_data_dir, x) for x in sorted(listdir(lr_data_dir)) if is_numpy_file(x)]


        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.num_low_receiver = num_low_receiver
        self.num_high_receiver = num_high_receiver
        # self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):

        numpy_adc = np.load(self.adc_filenames[index])
        # print(numpy_adc[0,0])
        adc_data = torch.from_numpy(numpy_adc) # D A R
        adc_data = adc_data.permute(2,0,1) # R D A
        # R D A 2
        adc_data = torch.view_as_real(adc_data).permute(3,2,0,1).type(torch.FloatTensor)/100 # devide by 100
        # 2 A R D

        cut_l = (adc_data.shape[1]-self.num_low_receiver)//2
        # print()
        adc_data_low = adc_data[:,cut_l:cut_l+self.num_low_receiver,:,:]
        # print(adc_data_low.shape)
        cut_l = (adc_data.shape[1]-self.num_high_receiver)//2
        # print()
        adc_data_high = adc_data[:,cut_l:cut_l+self.num_high_receiver,:,:]
        # print(adc_data_low.shape)
        

        # hr_data = complexTo2Channels(np.load(self.hr_filenames[index]))
        # hr_image = torch.from_numpy(hr_data.transpose(2,0,1)).type(torch.FloatTensor)/10

        # lr_data = complexTo2Channels(np.load(self.lr_filenames[index]))
        # lr_image = torch.from_numpy(lr_data.transpose(2,0,1)).type(torch.FloatTensor)/10

        

        return adc_data_high, adc_data_low, 2, 2
        return adc_data_high, adc_data_low, lr_image[None], hr_image[None]
        # return adc_data, adc_data_low, lr_image[None], hr_image[None]

    def __len__(self):
        return len(self.adc_filenames)

class ValDatasetFromFolder_radar3D(Dataset):
    def __init__(self,  hr_data_dir, lr_data_dir, upscale_factor, index_list=None):
        super().__init__()
        self.upscale_factor = upscale_factor
        # self.image_filenames = [join(hr_data_dir, x) for x in listdir(hr_data_dir) if is_image_file(x)]
        if index_list != None:
            self.hr_filenames = [x for x in sorted(glob(hr_data_dir)) if is_numpy_file(x)]
            self.lr_filenames = [x for x in sorted(glob(lr_data_dir)) if is_numpy_file(x)]
            
        else:
            self.hr_filenames = [join(hr_data_dir, x) for x in sorted(listdir(hr_data_dir)) if is_numpy_file(x)]
            self.lr_filenames = [join(lr_data_dir, x) for x in sorted(listdir(lr_data_dir)) if is_numpy_file(x)]

        # import pdb
        # pdb.set_trace()
        # self.hr_filenames = [join(hr_data_dir, x) for x in sorted(listdir(hr_data_dir)) if is_numpy_file(x)]
        # self.lr_filenames = [join(lr_data_dir, x) for x in sorted(listdir(lr_data_dir)) if is_numpy_file(x)]

    def __getitem__(self, index):
        
        hr_data = (np.load(self.hr_filenames[index]))
        lr_data = (np.load(self.lr_filenames[index]))

        hr_data = complexTo2Channels(hr_data)
        hr_image = torch.from_numpy(hr_data.transpose(2,0,1)).type(torch.FloatTensor)/10

        lr_data = complexTo2Channels(lr_data)
        lr_image = torch.from_numpy(lr_data.transpose(2,0,1)).type(torch.FloatTensor)/10


        return (lr_image[None]).type(torch.FloatTensor), (lr_image[None]).type(torch.FloatTensor), (hr_image[None]).type(torch.FloatTensor)

    def __len__(self):
        return len(self.hr_filenames)

class ValDatasetFromFolder_radar3D_adc(Dataset):
    def __init__(self, adc_data_dir, num_low_receiver, hr_data_dir, lr_data_dir, upscale_factor, index_list=None, num_high_receiver=12):
        super().__init__()
        self.upscale_factor = upscale_factor
        # self.image_filenames = [join(hr_data_dir, x) for x in listdir(hr_data_dir) if is_image_file(x)]
        if index_list != None:
            self.adc_filenames = [x for x in sorted(glob(adc_data_dir+'*')) if is_numpy_file(x)]
            # self.hr_filenames = [x for x in sorted(glob(hr_data_dir)) if is_numpy_file(x)]
            # self.lr_filenames = [x for x in sorted(glob(lr_data_dir)) if is_numpy_file(x)]
            print(len(self.adc_filenames))
            
        else:
            raise NotImplementedError
            # self.hr_filenames = [join(hr_data_dir, x) for x in sorted(listdir(hr_data_dir)) if is_numpy_file(x)]
            # self.lr_filenames = [join(lr_data_dir, x) for x in sorted(listdir(lr_data_dir)) if is_numpy_file(x)]

        self.num_low_receiver = num_low_receiver
        self.num_high_receiver = num_high_receiver

    def __getitem__(self, index):

        numpy_adc = np.load(self.adc_filenames[index])
        # print(numpy_adc[0,0])
        adc_data = torch.from_numpy(numpy_adc) # DAR
        adc_data = adc_data.permute(2,0,1) # RDA
        adc_sig_gt = adc_data.permute(0,2,1) # RAD
        adc_data = torch.view_as_real(adc_data).permute(3,2,0,1).type(torch.FloatTensor)/100 # devide by 100

        # 2 A R D

        cut_l = (adc_data.shape[1]-self.num_low_receiver)//2
        # print()
        adc_data_low = adc_data[:,cut_l:cut_l+self.num_low_receiver,:,:]

        cut_l = (adc_data.shape[1]-self.num_high_receiver)//2
        # print()
        adc_data_high = adc_data[:,cut_l:cut_l+self.num_high_receiver,:,:]
        # print(adc_data_low.shape)
        
        # hr_data = complexTo2Channels(np.load(self.hr_filenames[index])) # RAD
        # hr_image = torch.from_numpy(hr_data.transpose(2,0,1)).type(torch.FloatTensor)/10 # DAR

        # lr_data = complexTo2Channels(np.load(self.lr_filenames[index]))
        # lr_image = torch.from_numpy(lr_data.transpose(2,0,1)).type(torch.FloatTensor)/10

        

        return adc_data_high, adc_data_low, 2, 2
        return adc_data_high, adc_data_low, lr_image[None], hr_image[None]
        return adc_data, adc_data_low, lr_image[None], hr_image[None]
        return (lr_image[None]).type(torch.FloatTensor), (lr_image[None]).type(torch.FloatTensor), (hr_image[None]).type(torch.FloatTensor)

    def __len__(self):
        return len(self.adc_filenames)

class TrainDatasetFromFolder_radar(Dataset):
    def __init__(self, hr_data_dir, lr_data_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder_radar, self).__init__()
        self.hr_filenames = [join(hr_data_dir, x) for x in sorted(listdir(hr_data_dir)) if is_numpy_file(x)]
        self.lr_filenames = [join(lr_data_dir, x) for x in sorted(listdir(lr_data_dir)) if is_numpy_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        # self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):

        # import pdb
        # pdb.set_trace()
        # hr_image = torch.from_numpy(np.load(self.hr_filenames[index]).transpose(2,0,1)).type(torch.FloatTensor)
        # lr_image = torch.from_numpy(np.load(self.lr_filenames[index]).transpose(2,0,1)).type(torch.FloatTensor)
        hr_image = torch.from_numpy(np.log2(np.load(self.hr_filenames[index]).transpose(2,0,1))).type(torch.FloatTensor)
        lr_image = torch.from_numpy(np.log2(np.load(self.lr_filenames[index]).transpose(2,0,1))).type(torch.FloatTensor)
        # hr_image = (hr_image*10-140)*9/255
        # lr_image = (lr_image*10-140)*9/255
        # hr_image = (hr_image*10-100)*7/255
        # lr_image = (lr_image*10-100)*7/255
        # hr_image = hr_image*10/255
        # lr_image = lr_image*10/255
        # hr_image = np.clip(hr_image*13,0,255)/255
        # lr_image = np.clip(lr_image*13,0,255)/255
        # hr_image = np.clip((hr_image-14)*10,-255,255)/255
        # lr_image = np.clip((lr_image-14)*10,-255,255)/255
        # out_img = np.clip(out_img,0,255)
        # lr_image = self.lr_transform(hr_image)
        hr_image = hr_image/13-1
        lr_image = lr_image/13-1
        return lr_image, hr_image

    def __len__(self):
        return len(self.hr_filenames)


class ValDatasetFromFolder_radar(Dataset):
    def __init__(self,  hr_data_dir, lr_data_dir, upscale_factor):
        super(ValDatasetFromFolder_radar, self).__init__()
        self.upscale_factor = upscale_factor
        # self.image_filenames = [join(hr_data_dir, x) for x in listdir(hr_data_dir) if is_image_file(x)]
        self.hr_filenames = [join(hr_data_dir, x) for x in sorted(listdir(hr_data_dir)) if is_numpy_file(x)]
        self.lr_filenames = [join(lr_data_dir, x) for x in sorted(listdir(lr_data_dir)) if is_numpy_file(x)]

    def __getitem__(self, index):
        # hr_image = Image.open(self.image_filenames[index])
        # hr_image = torch.from_numpy(np.load(self.hr_filenames[index]).transpose(2,0,1))
        # lr_image = torch.from_numpy(np.load(self.lr_filenames[index]).transpose(2,0,1))

        hr_image_ = torch.from_numpy(np.log2(np.load(self.hr_filenames[index]).transpose(2,0,1)))
        lr_image_ = torch.from_numpy(np.log2(np.load(self.lr_filenames[index]).transpose(2,0,1)))
        # hr_image = (hr_image*10)*9/255
        # lr_image = (lr_image*10)*9/255
        # hr_image = hr_image*10/255
        # lr_image = lr_image*10/255
        # hr_image = np.clip(hr_image*13,0,255)/255
        # lr_image = np.clip(lr_image*13,0,255)/255
        # hr_image = np.clip((hr_image-14)*10,-255,255)/255
        # lr_image = np.clip((lr_image-14)*10,-255,255)/255
        hr_image = hr_image_/13-1
        lr_image = lr_image_/13-1




        # w, h = hr_image.size
        # crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        # lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        # hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)

        # hr_image = CenterCrop(crop_size)(hr_image)
        # lr_image = CenterCrop(crop_size)(lr_image)
        # lr_image = lr_scale(hr_image)

        # hr_restore_img = hr_scale(lr_image)
        return (lr_image).type(torch.FloatTensor), (lr_image_).type(torch.FloatTensor), (hr_image_).type(torch.FloatTensor)

    def __len__(self):
        return len(self.hr_filenames)

class ValDatasetFromFolder(Dataset):
    def __init__(self,  hr_data_dir, lr_data_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        # self.image_filenames = [join(hr_data_dir, x) for x in listdir(hr_data_dir) if is_image_file(x)]
        self.hr_filenames = [join(hr_data_dir, x) for x in sorted(listdir(hr_data_dir)) if is_image_file(x)]
        self.lr_filenames = [join(lr_data_dir, x) for x in sorted(listdir(lr_data_dir)) if is_image_file(x)]

    def __getitem__(self, index):
        # hr_image = Image.open(self.image_filenames[index])
        hr_image = Image.open(self.hr_filenames[index]).convert('RGB')
        lr_image = Image.open(self.lr_filenames[index]).convert('RGB')
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)

        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = CenterCrop(crop_size)(lr_image)
        lr_image = lr_scale(hr_image)

        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.hr_filenames)

# class ValDatasetFromFolder(Dataset):
#     def __init__(self, dataset_dir, upscale_factor):
#         super(ValDatasetFromFolder, self).__init__()
#         self.upscale_factor = upscale_factor
#         self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

#     def __getitem__(self, index):
#         hr_image = Image.open(self.image_filenames[index])
#         w, h = hr_image.size
#         crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
#         lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
#         hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
#         hr_image = CenterCrop(crop_size)(hr_image)
#         lr_image = lr_scale(hr_image)
#         hr_restore_img = hr_scale(lr_image)
#         return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

#     def __len__(self):
#         return len(self.image_filenames)

class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
        self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)
