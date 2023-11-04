# Radar 3D SR

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- PyTorch
```
conda install pytorch torchvision -c pytorch
```
- Other dependencies
```
pip install -r requirements.txt
```

## Datasets


## Usage

### Preprocess the data
To faciliate the training of the models, we first turn all of the binary files to numpy format and save into one directory:

```
python process_adc_bin2np.py
    --raw_data radar_bin/
    --output radar_numpy/
```

You will need to split the dataset `radar_numpy` into `radar_numpy_train` and `radar_numpy_test` 

### Train and validation

## ADC-SR
To train our pipeline of ADC-SR:


```
python train_ADC_SR.py 
    --output training_results/ADC_SR/
    --train_data radar_numpy_train/
    --val_data radar_numpy_test/
    --model_name aaa.pth

optional arguments:
--low_Azimuth              <low resolution>
[default value is 4](choices:[4, 8])
--high_Azimuth              <high resolution>
[default value is 12]]
--num_epochs                 <train epoch number>
[default value is 100]
--model_name     <resume model path>
```

The output val super resolution images are on `training_results/ADC_SR/` directory.


## RAD-SR
To train our pipeline of RAD-SR:


```
python train_RAD_SR.py 
    --output training_results/RAD_SR/
    --train_data radar_numpy_train/
    --val_data radar_numpy_test/

optional arguments:
--low_Azimuth              <low resolution>
[default value is 4](choices:[4, 8])
--high_Azimuth              <high resolution>
[default value is 12]]
--num_epochs                 <train epoch number>
[default value is 100]
--model_name     <resume model path>
```

The output validation super resolution images are on `training_results/RAD_SR/` directory.


## Hybrid-SR

To train our pipeline of Hybrid-SR:


```
python train_Hybrid_SR.py 
    --output training_results/Hybrid_SR/
    --train_data radar_numpy_train/
    --val_data radar_numpy_test/

optional arguments:
--low_Azimuth              <low resolution>
[default value is 4](choices:[4, 8])
--high_Azimuth              <high resolution>
[default value is 12]]
--num_epochs                 <train epoch number>
[default value is 100]
--model_g_name     <resume ADC-SR model path>
--model_r_name     <resume RAD-SR model path>
```

The output validation super resolution images are on `training_results/Hybrid_SR/` directory.




