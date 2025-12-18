# model structure
hyperparameter:
1. vocal_size: the number of character types in the read text
2. d_model: the length of embedding dimension
3. max_len: the maximum sequence length
4. n_heads: the number of dividing heads
5. hidden_dim: the number of extended hidden layer dimensions in FFN 
6. dropout: the proportion of randomly discarded units

- embedding layer
function: generate embedding vectors for every token
dimensional change: [batch_size, max_len]-> [batch_size, max_len, d_model]

- positional encoding
function: generate positional vectors and add to embedding vectors
- dimensional change: unchanged

- multi-head attention
function:
  1. pass the input data into the fully connected layer to obtain query, key, value
  2. divide query, key, value in the embedding vector dimension
  3. get attention scores by attention equations
  4. merge multiple heads and pass the result into a fully connected layer
dimensional change: [batch_size, max_len, d_model]->[batch_size, n_heads, max_len, head_dim]->[batch_size, max_len, d_model]

- add & norm
function: Perform residual connection and normalization on input data, to improve model's ability
dimensional change: unchanged

- FFN
function: 
  1. Enhance dimensions to uncover more data features
  2. Further reduce dimensions, remove redundant information, and maintain consistency between input and output
dimensional change: [batch_size, max_len, d_model]->[batch_size, max_Len, hidden_dim]->[batch_size, max_len, d_model]

- add & norm
function: Perform residual connection and normalization on input data, to improve model's ability
dimensional change: unchanged

- lm_head
function: mapping feature dimension d_model to vocal_size, to get the probability distribution of every token
dimensional change: [batch_size, max_len, d_model]->[batch_size, max_len, vocal_size]


# project structure
- data: save the data for training
- result: save the model and the training result during training process
- generate.py: test the effect of trained model
- training.py: handle the input data and train the model
- transformer.py: mini-transformer code