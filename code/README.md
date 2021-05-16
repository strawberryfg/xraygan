# Clean Code

*This is the final code for "Machine Learning - CSCI-GA.2565-001" Spring 2021 at NYU Courant.* 

Joy Chen, Qingfu Wan

----
----

## Conditional GAN

----
----

## Unconditional Score-guided GAN

----

### Train


1. At the beginning of `unconditionalGAN.py`, specify hyper-parameters and configuration parameters $$e.g.$$ `root_dir`,  learning rates,  loss weights, paths to store the weights, how many iterations per displaying... Specify used classes in *usable_label_arr*.
2. Put VGG weights `vgg_conv.pth` under`model_dir`.
3. Preprocessing:

   a. Use `../full_code/separate_image_index.py` to derive image indices from `data_entry.csv`.
   
   b. Use `../full_code/separate_labels.py` to get labels.
   
   c. Put the original `train_val_list.txt` and `test_list.txt` from the dataset as well as those from *step* **a.** and **b.** under `root_dir`. And check the paths $$e.g.$$ `image_index_list_file`.
   
   
4. Unzip the NIH Chest X-ray 14 dataset under `root_dir`: $e.g.$ `root_dir/images_0XX/images/` (XX={01, 02, ..., 12})

5. **Start training~~~!**

```
	python unconditionalGAN.py
``` 

* More options (*non-exhaustive*):
	* `resume_training`: 
	   * load D(discriminator),  G(generator) and C(classifier) from a previous checkpoint `model_path`.
	   * load D and G from `model_path_gan` and load from `model_path_cla`. 
	* `save_gan_per_iters`: save GANerated images on-the-fly
	* `show_test_classifier_acc_per_iters`: show intermediate test performances
	* `save_per_samples`: save a checkpoint per this number iterations



----




----

### Test

----
