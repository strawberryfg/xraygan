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
    
    
	| Pathology | Id   |  Acc (%)  | 
	|:-:|:-:|:-:|
	| Atelectasis     | 0 | 0.25 |
	| Cardiomegaly | 1 | 44.59  |
	| Effusion | 2 | 5.41 |
	| Infiltration | 3 | 0.00 |
	| Mass | 4 | 0.45 |
	| Nodule | 5 | 0.22 |
	| Pneumonia | 6 | 21.59  |
	| Pneumothorax | 7 | 0.95 |
	| Consolidation | 8 | 7.92  |
	| Edema | 9 | 69.26 | 
	| Emphysema | 10 |  14.19 |
	| Fibrosis | 11 | 19.43 |
	| Pleural Thickening | 12 | 15.21  |
	| Hernia | 13 | 22.22 |


 
 
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


        a) 

	| Pathology | Id   |  Acc (%)  | 
	|:-:|:-:|:-:|
	| Atelectasis     | 0 | 0.00 |
	| Cardiomegaly | 1 | 6.69  |
	| Effusion | 2 | 0.09 |
	| Infiltration | 3 | 0.00 |
	| Mass | 4 | 0.00 |
	| Nodule | 5 | 0.00 |
	| Pneumonia | 6 | 48.86  |
	| Pneumothorax | 7 | 0.00 |
	| Consolidation | 8 | 0.00  |
	| Edema | 9 | 40.69 | 
	| Emphysema | 10 |  0.99 |
	| Fibrosis | 11 | 24.00 |
	| Pleural Thickening | 12 | 0.00  |
	| Hernia | 13 | 57.78 |

   	Th = 0.8
	```
		python train_allclasses_thresh0.8.py
	```

	
	| Pathology | Id   |  Acc (%)  | 
	|:-:|:-:|:-:|
	| Atelectasis     | 0 | 0.00 |
	| Cardiomegaly | 1 | 4.14  |
	| Effusion | 2 | 0.00 |
	| Infiltration | 3 | 0.00 |
	| Mass | 4 | 0.00 |
	| Nodule | 5 | 0.00 |
	| Pneumonia | 6 | 39.77  |
	| Pneumothorax | 7 | 0.00 |
	| Consolidation | 8 | 0.21  |
	| Edema | 9 | 51.52 | 
	| Emphysema | 10 |  12.87 |
	| Fibrosis | 11 | 10.29 |
	| Pleural Thickening | 12 | 0.00  |
	| Hernia | 13 | 66.67 |
	
	

### Notes
