# GAN (GENERATIVE ADVERSARIAL NETWORKS)
## [DCGAN](https://arxiv.org/pdf/1511.06434.pdf) (DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS)
### Architecture
![Capture d’écran 2023-11-22 à 02 56 30](https://github.com/nhs2828/GANs/assets/78078713/93e94a38-7512-4cea-81c2-bc12ea7187cc)
*Architecture of the network*

This network is composed of 2 networks: the discriminator, whose role is to distingue real images and generated images, and the generator, whose role is to generate images from noises.
The principle of the loss is: the loss of the discriminator is BCE, real images would be class 1, and fake images would be class 0 as we want to distingue authentic images and fake ones. On the other hand, the loss of the generator is also BCE, as we want the generated images to be good enough to fool the discriminator, their labels are 1. The loss total is the mean of these 2 losses.
### Dataset
This implementation uses [cat_face](https://www.kaggle.com/datasets/spandan2/cats-faces-64x64-for-generative-models/code) dataset, in this dataset, the images are already in size 64x64 and centered at the faces, which improves the performance of the model.
### Results
