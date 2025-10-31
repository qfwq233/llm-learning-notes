## 🧪 experiment task
text classification

## 🎯 experiment destination
- understand the implementation of various neural networks
- compare the differences between different neural networks

##  📂 documents explanation
- code
  - text_cls_base.py: code for implementing text classification based on CNN
  - text_cls_lstm.py: code for implementing text classification based on LSTM
  - text_cls_transformer.py: code for implementing text classification based on TransformerEncoder
- log/ : the log in the process of training data and visualization results

##  📝 experiment Procedure
**level 1**: code based on CNN
Implement the overall process of data download, processing, training, and testing
**level 2**: code based on LSTM
Reconstruct LSTM neural network and replace CNN
**level 3**: code based on Transformer
Reconstruct Transformer and replace CNN

## ⚔️ result comparison
- CNN
In CNN, due to the direct extraction of n-gram local patterns by convolutional kernels and the short gradient path, convergence is faster and better performance can be achieved in fewer epochs
- LSTM
In Long Short-Term Memory (LSTM), due to the presence of multiple gating units within its structure, it has more parameters than Convolutional Neural Networks (CNN) and a longer gradient propagation path. 
This results in slow gradient updates, requiring a greater number of epochs to learn long-term dependencies. However, when the LSTM training duration is sufficiently long, its performance will gradually catch up with and even surpass that of CNN.
- Transformer
Training a Transformer from scratch on a medium-sized dataset, it is difficult for its performance to surpass that of neural networks such as CNNs.
 

