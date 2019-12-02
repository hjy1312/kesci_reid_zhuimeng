the training of Resnext101 include 2 steps: preliminary training and finetune.
the pretrained weights of Resnext101 on Imagenet is from the pytorch website.

1.pre-train:
./model includes the definition of the model and the initialization methods of its weight;
./utils includess the triplet sampling methods of the data loading,random erasing methods and the relevent training loss.
(1)./pre-train/offset_augment.py: augment the dataset by cropping;
(2)./train.py: pretrain the model for 80 epochs with PCB loss, global classification loss and triplet loss
to train the network, just run run.sh.

2.finetune:
in this stage, we set the probability of random erasing to 0, then we finetune the pretrained network for 60 epochs with PCB loss, global classification loss, triplet loss and focal loss.
the experimental setting details can be found at train.py.

