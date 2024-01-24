
<image src="./images/info-readme-architecture.png" alt="" height="500" wight="900"/>

# Vision Transfomer
The *Vision Transformer (ViT)* [paper](https://arxiv.org/abs/2010.11929) introduces a groundbreaking approach to computer vision tasks by leveraging the power of transformers, originally designed for natural language processing. Authored by Alexey Dosovitskiy et al., the paper challenges traditional convolutional neural networks (CNNs) in image classification and demonstrates the effectiveness of transformer architectures in capturing long-range dependencies within images.

ViT marks a departure from the conventional CNN-based methodologies that dominate the field of computer vision. By casting image processing as a sequence-to-sequence task, ViT transforms images into sequences of tokens, allowing transformers to be applied directly to the spatial information present in the data. This novel approach not only simplifies the architectural design but also showcases the versatility of transformers beyond text-based tasks.

In this introduction, we will delve into the key motivations behind the development of Vision Transformer, the core architectural components, and the notable experimental results that position ViT as a pioneering model in the realm of image understanding. As we navigate through the paper, we will uncover the unique contributions that ViT brings to the table, shedding light on its potential implications for the future of computer vision.



From the diagram is guided by four equation from the below table that shows how one image is passed through out the architecture during training and inference.


# Quick Start Example

``` python 
from modular_functions import vit

model, results = vit.train(
        # model=created_model, # DEFAULT pretrained 16 base model,
        # transforms=created_transform # DEFAULT  transform from the pretrained model
        # batch_size=created_batchsize # DEFAULT batch_size = 32
        train_dir=train_dir,
        test_dir=test_dir,
        device="mps" # <- It is trained on silicon chip but can be changed to "gpu" 
        )

```

