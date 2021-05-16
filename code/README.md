# Clean Code

*This is the final code for "Machine Learning - CSCI-GA.2565-001" Spring 2021 at NYU Courant.* 

----
----

## Conditional GAN

----
----

## Unconditional Score-guided GAN

----

### Train
1. At the beginning of `unconditionalGAN.py`, specify hyper-parameters and configuration parameters $e.g.$ `root_dir`,  loss weights, paths to store the weights, where to save GANerated images on-the-fly, how many iterations before testing/displaying... Specify used classes in *usable_label_arr*.
2. Put VGG weights `vgg_conv.pth` under`model_dir`.
3. Preprocessing:
   a. Use `../full_code/separate_image_index.py` to derive image indices from `data_entry.csv`.
   b. Use `../full_code/separate_labels.py` to get labels.
   c. Put the original `train_val_list.txt` and `test_list.txt` from the dataset as well as those from *step* **a.** and **b.** under `root_dir`. And check the paths $e.g.$ `image_index_list_file`.
4. Unzip the NIH Chest X-ray 14 dataset under `root_dir`: $e.g.$ `root_dir/images_0XX/images/` (XX={01, 02, ..., 12})


----




----

### Test

----
