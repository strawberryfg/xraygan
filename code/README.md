# Code


## Supported features:
    Randomly fetching data from SSD; logging train/val loss/acc.

## Run on Colab (Notebook friendly)

* Copy this code and directly run. No import from other folders etc. All networks are defined within this file.

## Run on local machines
1. Specify hyper-parameters in **#5** section. Set root directories (*mount on Google Drive if using Colab and use "/content/"-like path*) in `root_dir`. The code used absolute path. 

2. Optionally set other parameters e.g. batch_size, how many iterations to save a checkpoint, how many iterations to save GAN images, how many iterations to output loss etc.

3. Create a `models/` folder to save checkpoints before running. Logs will be automatically saved to `logs/`.

4. Images are sampled randomly on-the-fly, so no pre-constructed dataset object. 



## Train 
1. Train EC-GAN with *128x128* size output (instead of *32x32* in paper) and ResNet-50 as the classifier.
```
	python train_apr18.py
```

This one shows training accuracy on real data.

2. Show test accuracy while training to guide the learning of GAN. Changed adversarial weight to *0.25*. **Included batch-wise Inception Score and Maximum Mean Discrepancy as part of loss.**

```
	python train_apr19.py
```

3. Train on Chest Xray 14 dataset 

    **One model for all classes**
    
	a. inception score (KL)
	
	b. maximum mean discrepancy (MMD)
	
	c. style transfer gram matrix loss (**Style Reconstruction Loss** in the **Perceptual** paper)
 
 
```
	python train_apr27.py
```


4. Train a vanilla Res50 on all classes 

    **One model for all classes only classification loss**
    
    Epoch 99 Per Class Acc 0.25 44.59 5.41 0.00 0.45 0.22 21.59 0.95 7.92 69.26 14.19 19.43 15.21 22.22  
 
 
```
	python train_all_classes_cla_vanilla.py
```

To test
```
	python test_all_classes_cla_vanilla.py
```

5. Train with pseudo labels (confidenceThresh = 0.7)

```
	python train_allclasses.py
```

    Epoch 105 Per Class Acc 0.00 6.69 0.09 0.00 0.00 0.00 48.86 0.00 0.00 40.69 0.99 24.00 0.00 57.78  

    Th = 0.8

### Notes
