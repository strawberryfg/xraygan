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
 

### Notes
