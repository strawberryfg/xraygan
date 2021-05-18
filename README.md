
# XRAY-GAN: Conditional and Unconditional Score-guided GAN for Chest X-ray Pathology Classification

Joy Chen, Qingfu Wan



*This is the repository for the final project of "Machine Learning - CSCI-GA.2565-001" Spring 2021 at NYU Courant.* 


----


# Pipelines

   ## CGAN


<p align="center">  
<img src="figs/cgan.gif" width="600" height="300" >  
</p> 

  ## UCGAN
  
  
<p align="center">  
<img src="figs/xraygan.gif" width="600" height="350">  
</p> 


----
# Features
----

- Conditioning the image generation on class labels for a better *equilibrium* in *imbalanced multi-modal* data.

- Metrics-turned-Trainable-Scores + Self-aligned Perceptual Style Scores for a better-learned *data distribution*.

- Built-in classifier as an *implicit* evaluator and an *explicit* learner.

- Label-aware *class-specific* generation + class-agnostic *homogenous* generation system.

- A foolproof *online self-annotation* mechanism compensating for insufficient data.

----
# Report
----

Report can be found here.

----
# General Structure
----

   ```
   ${ROOT}
   +-- experiments
   +-- figs   
   +-- code
   +-- full_code
   +-- README.md
   ```




# Clean Code

   
   `${ROOT}/code/`
   
   
# The Complete Set of Code

   
   `${ROOT}/full_code/`



----
# Environment
----

- PyTorch
- Keras


----
# GANerated Images
----
   
**Left**: Real Data; **Right**: Generated Data


## CGAN

Clear indication of ribs, spine, and heart shading.

<p align="center">  
<img src="figs/cgan-image-comparison.gif" width="550" height="260">  
</p> 

## UCGAN

Clear spines, lungs, and varied contrast \& color gradient.


<p align="center">  
<img src="figs/ucganimgs.gif" width="550" height="310">  
</p> 




----
# Enjoy!~
----
   
