# CP-VAE
(Cell Profiler Variational Auto Encoder)

A variational autoencoder with morphological profiling for fluorescence images of MCF-7 breast cancer cells.

## Summary
* VAE trained on small portion of <a href="https://bbbc.broadinstitute.org/BBBC021">BBBC021</a> dataset from broad institute. Also outputs predictions for morphological features from on the dataset (generated with <a href="https://cellprofiler.org/">CellProfiler</a>).
* VAE based on [1]
* P(x|z) modelled with continuous bernoulli [2]
* Encoder and decoder are ResNets [3]. (I have found that ResNets perform consistently better than vanilla CNNs on image tasks)
* Downsampling in encoder with strided convolutions.
* Upcaling in decoder done by nearest neighbour interpolation upscaling followed by conv rather than a 1 step transpose conv, see [4]
* kl loss (ELBO) scaled with beta parameter as in b-VAE [5] and uses kl annealing.
* Cosine annealing of learning rate
* Adam optimiser
* latent space of 64(8\*8) or 256(16\*16) dimensions (simply beacuse it's easy to scale up to 128\*128\*3, the size of the images)
* results shown from 64 dim model


## Future work
* Train on more data. The model was trained on only about 10% of the total dataset.
* More tuning of Hyperparameters to better balance the three losses.
* Encode treatments. The cells in the images have been treated with various small molecules. Would be interesting to add this to predict/visualise effects of drugs on the cells.

## Refs
* [1] https://arxiv.org/abs/1312.6114
* [2] https://arxiv.org/abs/1907.06845
* [3] https://arxiv.org/abs/1512.03385
* [4] https://distill.pub/2016/deconv-checkerboard/
* [5] https://openreview.net/pdf?id=Sy2fzU9gl

## Results
### Reconstructions
<img src="https://raw.githubusercontent.com/RodMcN/CP-VAE/master/imgs/rec.png">

### Generated
Generated images created by sampling from normal distribution N(0, 1) of same dim as latent and passing to values to decoder
<img src="https://raw.githubusercontent.com/RodMcN/CP-VAE/master/imgs/gen.png">

### Morphological feature prediction
Covariance of ground truth measurements

<img src="https://raw.githubusercontent.com/RodMcN/CP-VAE/master/imgs/pred_cov.png">

Link to list of measurements

per-measurement mean absolute error predicted

per-measurement mean absolute error generated

### Latent space
based on mu-values output from encoder on single images

#### Principle Component Analysis
<img src="https://raw.githubusercontent.com/RodMcN/CP-VAE/master/imgs/pca_plot.png">

Explained variance: PC1: 8.28253536, PC2: 5.38001632

<img src="https://raw.githubusercontent.com/RodMcN/CP-VAE/master/imgs/pca_gen.png">

#### U-MAP of embeddings
<img src="https://raw.githubusercontent.com/RodMcN/CP-VAE/master/imgs/umap.png">


#### Probability distributions in latent space
TODO