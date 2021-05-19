# Code


*This is the complete set of codes* 


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


2. Show test accuracy while training to guide the learning of GAN. Changed adversarial weight to *0.25*. 
	
	**Included batch-wise Inception Score and Maximum Mean Discrepancy as part of loss.**

	```
		python train_apr19.py
	```

3. Train on Chest X-ray 14 dataset 

    **One model for all classes w/ all the losses (Full Pipeline)**
    
	    a. inception score (KL)
	
	    b. maximum mean discrepancy (MMD)
	
	    c. style transfer gram matrix loss (**Style Reconstruction Loss** in the **Perceptual** paper)
	    
	  ```
		python train_apr27.py
	  ```


4. *Baseline* Train a vanilla Res50 on all classes 

    **One model for all classes w/ classification loss only**
    
    
	| Pathology | Class Id   |  Acc (%)  | AUROC |
	|:-:|:-:|:-:|:-:|
	| Pneumonia | 7 | -  | 0.63 |
	| Edema | 10 | - | 0.75 |
	| Hernia | 14 | - | 0.88 |


 
 
	```
		python train_allclasses_cla_vanilla.py
	```

	To test (similarly for the following with "test" prefixes)
	```
		python test_allclasses_cla_vanilla.py
	```
	
	  a) To limit the scope to three classes (*Pneumonia*, *Edema*, *Hernia*), use the following 
	  
		```
			python train_allclasses_cla_vanilla.py		
		```


5. *Ablation 1* Train with pseudo labels 

    **Pseudo label confidence threshold**

    ```		
		python train_allclasses_wis_mmd_nst_threshP.py
	``` 
	
	where P is the threshold
    
    a) Thresh = 0.6

	| Pathology | Class Id   |  Acc (%)  | AUROC |
	|:-:|:-:|:-:|:-:|
	| Pneumonia | 7 | -  | 0.57 |
	| Edema | 10 | - | 0.82 |
	| Hernia | 14 | - | 0.94 |
	
    b) Thresh = 0.7

	| Pathology | Class Id   |  Acc (%)  | AUROC |
	|:-:|:-:|:-:|:-:|
	| Pneumonia | 7 | -  | 0.56 |
	| Edema | 10 | - | 0.82 |
	| Hernia | 14 | - | 0.93 |

    **c) Thresh = 0.8**
	
	| Pathology | Class Id   |  Acc (%)  | AUROC |
	|:-:|:-:|:-:|:-:|
	| Pneumonia | 7 | -  | 0.69 |
	| Edema | 10 | - |  0.82 |
	| Hernia | 14 | - | 0.90 |
	
      d) Thresh = 0.9
	
	| Pathology | Class Id   |  Acc (%)  |  AUROC |
	|:-:|:-:|:-:|:-:|
	| Pneumonia | 7 | -  | 0.54 |
	| Edema | 10 | - | 0.81 |
	| Hernia | 14 | - | 0.92 |



6. *Ablation 2* Train with simplest GAN losses + IS loss + MMD loss + NST loss 

    *confidence threshold = 0.75* for this ablation.
   
	- + Simple (ordinary discriminator, generator and classification losses)
	
	```
		python train_allclasses_wo.py
	```

	| Pathology | Class Id   |  Acc (%)  |  AUROC |
	|:-:|:-:|:-:|:-:|
	| Pneumonia | 7 | -  | 0.69 |
	| Edema | 10 | - | 0.78 |
	| Hernia | 14 | - | 0.82 |
	
	- + IS
	
	```
		python train_allclasses_wis_only.py
	```

	| Pathology | Class Id   |  Acc (%)  |  AUROC |
	|:-:|:-:|:-:|:-:|
	| Pneumonia | 7 | -  | 0.68 |
	| Edema | 10 | - | 0.80 |
	| Hernia | 14 | - | 0.91 |
	
	- + IS + MMD
	
	```
		python train_allclasses_wis_mmd.py
	```
	
	| Pathology | Class Id   |  Acc (%)  |  AUROC |
	|:-:|:-:|:-:|:-:|
	| Pneumonia | 7 | -  | 0.67 |
	| Edema | 10 | - | 0.82 |
	| Hernia | 14 | - | 0.94 |
	
	- + IS + MMD + NST 
	
	```
		python train_allclasses_wis_mmd_nst.py
	```
	
	| Pathology | Class Id   |  Acc (%)  |  AUROC |
	|:-:|:-:|:-:|:-:|
	| Pneumonia | 7 | -  | 0.70 |
	| Edema | 10 | - | 0.83 |
	| Hernia | 14 | - | 0.92 |


7. *Ablation 3* Train with few real samples for one class (Edema). Vary #(real samples)

      *confidence threshold = 0.75* for this ablation too.
      
	- + 
	
	```
		python train_pneuede_wis_mmd_nst.py
	```
	
	
	```
		python train_allclasses_wis_mmd_nst_thresh0.75_fake_cla_trainer_only.py.py
	```
	
      i. train only on fake images
      
	| Pathology | Class Id   |  Acc (%)  |  AUROC |
	|:-:|:-:|:-:|:-:|
	| Pneumonia | 7 | -  | - |
	| Edema | 10 | - | 0.53 |
	| Hernia | 14 | - | - |
	

      ii. 20 samples for Edema (both for GAN and for classification head)
      
	| Pathology | Class Id   |  Acc (%)  |  AUROC |
	|:-:|:-:|:-:|:-:|
	| Pneumonia | 7 | -  | - |
	| Edema | 10 | - | 0.52 |
	| Hernia | 14 | - | - |
	
      iii. 30 samples for Edema
      
	| Pathology | Class Id   |  Acc (%)  |  AUROC |
	|:-:|:-:|:-:|:-:|
	| Pneumonia | 7 | -  | - |
	| Edema | 10 | - | 0.62 |
	| Hernia | 14 | - | - |
	
      iv. 40	
        | Pathology | Class Id   |  Acc (%)  |  AUROC |
	|:-:|:-:|:-:|:-:|
	| Pneumonia | 7 | -  | - |
	| Edema | 10 | - | 0.71 |
	| Hernia | 14 | - | - |
	
      v. 100	
        | Pathology | Class Id   |  Acc (%)  |  AUROC |
	|:-:|:-:|:-:|:-:|
	| Pneumonia | 7 | -  | - |
	| Edema | 10 | - | 0.71 |
	| Hernia | 14 | - | - |



### Notes
