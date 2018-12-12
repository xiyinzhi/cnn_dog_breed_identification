## Dog Breed Identification


### Description

Who's a good dog? Who likes ear scratches? Well, it seems those fancy deep neural networks don't have all the answers. However, maybe they can answer that ubiquitous question we all ask when meeting a four-legged stranger: what kind of good pup is that?

In this playground competition, you are provided a strictly canine subset of ImageNet in order to practice fine-grained image categorization. How well you can tell your Norfolk Terriers from your Norwich Terriers? With 120 breeds of dogs and a limited number training images per class, you might find the problem more, err, ruff than you anticipated.

### Acknowledgments

We extend our gratitude to the creators of the Stanford Dogs Dataset for making this competition possible: Aditya Khosla, Nityananda Jayadevaprakash, Bangpeng Yao, and Fei-Fei Li.

### Evaluation

Submissions are evaluated on Multi Class Log Loss between the predicted probability and the observed target.

### Challenges

1. Fine-grained classification problems 
2. Data Imbalance & Small Amount                   
3. Train set is almost as big as test set

### Framework & Machine
PyTorch 0.20

EC2(Tesla K80 *1)

### Our work

1.Data balanced & data augmentation

2.Use dog crops/Faster-RCNN to pre-train model of PASCAL_VOC(21 classes), to classify dogs from other animals

3.Train a models with 6 layers:

vgg16 + resnet50 + resnet101 + resnet152 + resnetinception_v2 + inception_v3

