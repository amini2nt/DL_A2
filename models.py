import torch 
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt

# NOTE ==============================================
#
# Fill in code for every method which has a TODO
#
# Your implementation should use the contract (inputs
# and outputs) given for each model, because that is 
# what the main script expects. If you modify the contract, 
# you must justify that choice, note it in your report, and notify the TAs 
# so that we run the correct code.
#
# You may modify the internals of the RNN and GRU classes
# as much as you like, except you must keep the methods
# in each (init_weights_uniform, init_hidden, and forward)
# Using nn.Module and "forward" tells torch which 
# parameters are involved in the forward pass, so that it
# can correctly (automatically) set up the backward pass.
#
# You should not modify the interals of the Transformer
# except where indicated to implement the multi-head
# attention. 


def clones(module, N):
    '''
    A helper function for producing N identical layers (each with their own parameters).
    
    inputs: 
        module: a pytorch nn.module
        N (int): the number of copies of that module to return

    returns:
        a ModuleList with the copies of the module (the ModuleList is itself also a module)
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

'''
# Problem 1
class RNN(nn.Module): # Implement a stacked vanilla RNN with Tanh nonlinearities.
  def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
    """
    emb_size:     The number of units in the input embeddings
    hidden_size:  The number of hidden units per layer
    seq_len:      The length of the input sequences
    vocab_size:   The number of tokens in the vocabulary (10,000 for Penn TreeBank)
    num_layers:   The depth of the stack (i.e. the number of hidden layers at 
                  each time-step)
    dp_keep_prob: The probability of *not* dropping out units in the 
                  non-recurrent connections.
                  Do not apply dropout on recurrent connections.
    """
    super(RNN, self).__init__()

    # TODO ========================
    # Initialization of the parameters of the recurrent and fc layers. 
    # Your implementation should support any number of stacked hidden layers 
    # (specified by num_layers), use an input embedding layer, and include fully
    # connected layers with dropout after each recurrent layer.
    # Note: you may use pytorch's nn.Linear, nn.Dropout, and nn.Embedding 
    # modules, but not recurrent modules.
    #
    # To create a variable number of parameter tensors and/or nn.Modules 
    # (for the stacked hidden layer), you may need to use nn.ModuleList or the 
    # provided clones function (as opposed to a regular python list), in order 
    # for Pytorch to recognize these parameters as belonging to this nn.Module 
    # and compute their gradients automatically. You're not obligated to use the
    # provided clones function.
    self.emb_size = emb_size
    self.hidden_size = hidden_size
    self.seq_len = seq_len
    self.batch_size = batch_size
    self.vocab_size = vocab_size
    self.num_layers = num_layers
    self.dp_keep_prob = dp_keep_prob

    self.embedding = nn.Embedding(self.vocab_size, self.emb_size)


    self.layer_list = []
    self.layer_sizes = []
    self.dropout_list = []
    self.layer_sizes.append(self.emb_size)
    for i in range(self.num_layers):
        self.layer_sizes.append(self.hidden_size)
    self.layer_sizes.append(self.vocab_size)

    for i in range(len(self.layer_sizes)-2):
        self.layer_list.append(nn.Linear(self.layer_sizes[i]+self.hidden_size, self.layer_sizes[i+1]))
    self.layer_list.append(nn.Linear(self.layer_sizes[-2], self.layer_sizes[-1]))

    for i in range(num_layers):
        self.dropout_list.append(nn.Dropout(1-self.dp_keep_prob))
    self.list_of_layer_modules = nn.ModuleList(self.layer_list)
    self.list_of_layer_dropouts = nn.ModuleList(self.dropout_list)
    self.input_dropout = nn.Dropout(1-self.dp_keep_prob)
    self.init_weights()
    if torch.cuda.is_available():
        self.device = torch.device("cuda") 
    else:
        self.device = torch.device("cpu")

  def init_weights(self):
    # TODO ========================
    # Initialize the embedding and output weights uniformly in the range [-0.1, 0.1]
    # and output biases to 0 (in place). The embeddings should not use a bias vector.
    # Initialize all other (i.e. recurrent and linear) weights AND biases uniformly 
    # in the range [-k, k] where k is the square root of 1/hidden_size
    embedding_weights = np.random.uniform(low=-0.1, high=0.1, size=(self.vocab_size, self.emb_size))
    self.embedding.weight.data.copy_(torch.from_numpy(embedding_weights))

    torch.nn.init.zeros_(self.layer_list[-1].bias)
    torch.nn.init.uniform_(self.layer_list[-1].weight, a=-0.1, b=0.1)


    rng = np.sqrt(1.0/self.hidden_size)
    for i in range(len(self.layer_list) -1 ):
        torch.nn.init.uniform_(self.layer_list[i].bias, a=-rng, b=rng)
        torch.nn.init.uniform_(self.layer_list[i].weight, a=-rng, b=rng)

    return


  def init_hidden(self):
    # TODO ========================
    # initialize the hidden states to zero
    """
    This is used for the first mini-batch in an epoch, only.
    
    return # a parameter tensor of shape (self.num_layers, self.batch_size, self.hidden_size)
    """
    return torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=self.device)
    
  def forward(self, inputs, hidden):
    # TODO ========================
    # Compute the forward pass, using nested python for loops.
    # The outer for loop should iterate over timesteps, and the 
    # inner for loop should iterate over hidden layers of the stack. 
    # 
    # Within these for loops, use the parameter tensors and/or nn.modules you 
    # created in __init__ to compute the recurrent updates according to the 
    # equations provided in the .tex of the assignment.
    #
    # Note that those equations are for a single hidden-layer RNN, not a stacked
    # RNN. For a stacked RNN, the hidden states of the l-th layer are used as 
    # inputs to to the {l+1}-st layer (taking the place of the input sequence).

    """
    Arguments:
        - inputs: A mini-batch of input sequences, composed of integers that 
                    represent the index of the current token(s) in the vocabulary.
                        shape: (seq_len, batch_size)
        - hidden: The initial hidden states for every layer of the stacked RNN.
                        shape: (num_layers, batch_size, hidden_size)
    
    Returns:
        - Logits for the softmax over output tokens at every time-step.
              **Do NOT apply softmax to the outputs!**
              Pytorch's CrossEntropyLoss function (applied in ptb-lm.py) does 
              this computation implicitly.
                    shape: (seq_len, batch_size, vocab_size)
        - The final hidden states for every layer of the stacked RNN.
              These will be used as the initial hidden states for all the 
              mini-batches in an epoch, except for the first, where the return 
              value of self.init_hidden will be used.
              See the repackage_hiddens function in ptb-lm.py for more details, 
              if you are curious.
                    shape: (num_layers, batch_size, hidden_size)
    """

    output_list = []
    prev_hidden = hidden
    for t in range(self.seq_len):
        input_t = inputs[t]

        new_hidden = []

        x_t = self.embedding(input_t)
        x_t = self.input_dropout(x_t)

        for i in range(self.num_layers):
            concatenated_input = torch.cat([x_t, prev_hidden[i]], dim=1)
            h_t = self.layer_list[i](concatenated_input)
            h_t = torch.tanh(h_t)
            h_t = self.dropout_list[i](h_t)
            x_t = h_t
            
            new_hidden.append(h_t)

        prev_hidden = new_hidden

        output = self.layer_list[-1](x_t)
        output_list.append(output)

    return torch.stack(output_list), torch.stack(prev_hidden)

  def generate(self, input, hidden, generated_seq_len):
    # TODO ========================
    # Compute the forward pass, as in the self.forward method (above).
    # You'll probably want to copy substantial portions of that code here.
    # 
    # We "seed" the generation by providing the first inputs.
    # Subsequent inputs are generated by sampling from the output distribution, 
    # as described in the tex (Problem 5.3)
    # Unlike for self.forward, you WILL need to apply the softmax activation 
    # function here in order to compute the parameters of the categorical 
    # distributions to be sampled from at each time-step.

    """
    Arguments:
        - input: A mini-batch of input tokens (NOT sequences!)
                        shape: (batch_size)
        - hidden: The initial hidden states for every layer of the stacked RNN.
                        shape: (num_layers, batch_size, hidden_size)
        - generated_seq_len: The length of the sequence to generate.
                       Note that this can be different than the length used 
                       for training (self.seq_len)
    Returns:
        - Sampled sequences of tokens
                    shape: (generated_seq_len, batch_size)
    """
   
    return samples
'''


class RNN_hidden_layer(nn.Module):
    def __init__(self, x_dim, h_dim, dp_keep_prob):
        super(RNN_hidden_layer, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim

        self.dropout = nn.Dropout(p=(1 - dp_keep_prob))
        self.W = nn.Linear(in_features=(x_dim + h_dim),
                           out_features=h_dim,
                           bias=True)
        self.tanh = nn.Tanh()

    def init_weights_uniform(self):
        nn.init.uniform_(self.W.weight.data,
                         a=-np.sqrt(1/self.h_dim), b=np.sqrt(1/self.h_dim))
        nn.init.uniform_(self.W.bias.data,
                         a=-np.sqrt(1/self.h_dim), b=np.sqrt(1/self.h_dim))

    def forward(self, x, h):
        x = self.dropout(x)
        l_in = torch.cat((x, h), dim=1)
        l_out = self.W(l_in)
        out = self.tanh(l_out)
        return out


class RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size,
                 num_layers, dp_keep_prob):
        """
        emb_size:     The number of units in the input embeddings
        hidden_size:  The number of hidden units per layer
        seq_len:      The length of the input sequences
        vocab_size:   The number of tokens in the vocabulary
                      (10,000 for Penn TreeBank)
        num_layers:   The depth of the stack (i.e. the number of hidden layers
                      at each time-step)
        dp_keep_prob: The probability of *not* dropping out units in the 
                      non-recurrent connections.
        """
        super(RNN, self).__init__()

        # Params
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dp_keep_prob = dp_keep_prob

        # Input Embedding layer
        self.emb_layer = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=emb_size)

        # Hidden layers as a MduleList
        self.hidden_layers = nn.ModuleList()

        # Hidden layers
        for i in range(num_layers):
            input_dim = emb_size if i == 0 else hidden_size
            self.hidden_layers.append(
                                    RNN_hidden_layer(x_dim=input_dim,
                                                     h_dim=hidden_size,
                                                     dp_keep_prob=dp_keep_prob)
                                    )

        # Out layer
        self.out_dropout = nn.Dropout(p=(1 - dp_keep_prob))
        self.out_layer = nn.Linear(in_features=hidden_size,
                                   out_features=vocab_size,
                                   bias=True)

        # Initialize all weights
        self.init_weights_uniform()

    def init_weights_uniform(self):
        # Initialize the embedding and output weights uniformly in the range
        # [-0.1, 0.1] and the embedding and output biases to 0 (in place).
        # Initialize all other (i.e. recurrent and linear) weights AND biases
        # uniformly in the range [-k, k] where k is the square root of
        # 1/hidden_size
        nn.init.uniform_(self.emb_layer.weight.data, a=-0.1, b=0.1)
        nn.init.uniform_(self.out_layer.weight.data, a=-0.1, b=0.1)
        nn.init.zeros_(self.out_layer.bias.data)
        for hidden_layer in self.hidden_layers:
            hidden_layer.init_weights_uniform()

    def init_hidden(self):
        # initialize the hidden states to zero
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)

    def forward(self, inputs, hidden):
        """
        Arguments:
            - inputs: A mini-batch of input sequences, composed of integers that
                      represent the index of the current token(s) in the
                      vocabulary.
                            shape: (seq_len, batch_size)
            - hidden: The initial hidden states for every layer of the stacked
                      RNN.
                            shape: (num_layers, batch_size, hidden_size)

        Returns:
            - Logits for the softmax over output tokens at every time-step.
                        shape: (seq_len, batch_size, vocab_size)
            - The final hidden states for every layer of the stacked RNN.
                        shape: (num_layers, batch_size, hidden_size)
        """

        # To save outputs at each time step
        logits = torch.zeros([self.seq_len, self.batch_size, self.vocab_size],
                             device=inputs.device)

        # Input to hidden layer - embedding of input
        emb_input = self.emb_layer(inputs)    # (seq_len, batch_size, emb_size)

        # For each time step
        for t in range(self.seq_len):

            # Input at this time step for each layer
            input_l = emb_input[t]      # (batch_size, emb_size)

            # Next hidden layer
            hidden_next_t = []

            # For each layer
            for l, h_layer in enumerate(self.hidden_layers):

                # Get hidden layer output
                h_layer_out_t = h_layer(input_l, hidden[l])

                # Input for next layer
                input_l = h_layer_out_t

                # Hidden state for next time step
                hidden_next_t.append(h_layer_out_t)

            # Stack next hidden layer
            hidden = torch.stack(hidden_next_t)

            # Get output at this time step
            h_layer_out_dropout = self.out_dropout(input_l)
            logits[t] = self.out_layer(h_layer_out_dropout)

        # Return logits: (seq_len, batch_size, vocab_size),
        #        hidden: (num_layers, batch_size, hidden_size)
        return logits, hidden

    def generate(self, input, hidden, generated_seq_len):
        # Compute the forward pass, as in the self.forward method (above).
        # You'll probably want to copy substantial portions of that code here.
        # 
        # We "seed" the generation by providing the first inputs.
        # Subsequent inputs are generated by sampling from the output
        # distribution, as described in the tex (Problem 5.3)
        # Unlike for self.forward, you WILL need to apply the softmax activation
        # function here in order to compute the parameters of the categorical 
        # distributions to be sampled from at each time-step.

        """
        Arguments:
            - input: A mini-batch of input tokens (NOT sequences!)
                            shape: (batch_size)
            - hidden: The initial hidden states for every layer of the stacked
                      RNN.
                            shape: (num_layers, batch_size, hidden_size)
            - generated_seq_len: The length of the sequence to generate.
                           Note that this can be different than the length used 
                           for training (self.seq_len)
        Returns:
            - Sampled sequences of tokens
                        shape: (generated_seq_len, batch_size)
        """

        # Input to hidden layer - embedding of input
        samples = input.view(1, -1)         # (1, batch_size)

        # Input to hidden layer - embedding of input
        emb_input = self.emb_layer(samples)    # (1, batch_size, emb_size)

        # For each time step
        for t in range(generated_seq_len):

            # Next hidden layer
            hidden_next_t = []

            # Input at this time step for each layer
            input_l = emb_input[0]      # (batch_size, emb_size)

            # For each layer
            for l, h_layer in enumerate(self.hidden_layers):

                # Get hidden layer output
                h_layer_out_t = h_layer(input_l, hidden[l])

                # Input for next layer
                input_l = h_layer_out_t

                # Hidden state for next time step
                hidden_next_t.append(h_layer_out_t)

            # Stack next hidden layer
            hidden = torch.stack(hidden_next_t)

            # Get output at this time step
            h_layer_out_dropout = self.out_dropout(input_l)
            logits = self.out_layer(h_layer_out_dropout).detach()
            probs = F.softmax(logits, dim=1)    # (batch_size, vocab_size)
            token_out = Categorical(probs=probs).sample()   # (batch_size)
            token_out = token_out.view(1, -1)               # (1, batch_size)

            # Append output to samples
            samples = torch.cat((samples, token_out), dim=0)

            # Make input to next time step
            emb_input = self.emb_layer(token_out)   # (1, batch_size, emb_size)

        return samples


# Problem 2
class GRU(nn.Module): # Implement a stacked GRU RNN
  """
  Follow the same instructions as for RNN (above), but use the equations for 
  GRU, not Vanilla RNN.
  """
  def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
    super(GRU, self).__init__()

    # TODO ========================


    self.emb_size = emb_size
    self.hidden_size = hidden_size
    self.seq_len = seq_len
    self.batch_size = batch_size
    self.vocab_size = vocab_size
    self.num_layers = num_layers
    self.dp_keep_prob = dp_keep_prob

    self.r_list = []
    self.z_list = []
    self.hp_list = []

    self.layer_sizes = []
    self.dropout_list = []
    self.layer_sizes.append(self.emb_size)
    for i in range(self.num_layers):
        self.layer_sizes.append(self.hidden_size)
    self.layer_sizes.append(self.vocab_size)

    for i in range(len(self.layer_sizes)-2):
        self.r_list.append(nn.Linear(self.layer_sizes[i]+self.hidden_size, self.layer_sizes[i+1]))
        self.z_list.append(nn.Linear(self.layer_sizes[i]+self.hidden_size, self.layer_sizes[i+1]))
        self.hp_list.append(nn.Linear(self.layer_sizes[i]+self.hidden_size, self.layer_sizes[i+1]))

    self.output_layer =  nn.Linear(self.layer_sizes[-2], self.layer_sizes[-1])

    for i in range(num_layers): 
        self.dropout_list.append(nn.Dropout(1-self.dp_keep_prob))
    
    self.list_of_r_modules = nn.ModuleList(self.r_list)
    self.list_of_z_modules = nn.ModuleList(self.z_list)
    self.list_of_hp_modules = nn.ModuleList(self.hp_list)
    

    self.list_of_layer_dropouts = nn.ModuleList(self.dropout_list)
    self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
    self.input_dropout = nn.Dropout(1-self.dp_keep_prob)
    self.init_weights()
    if torch.cuda.is_available():
        self.device = torch.device("cuda") 
    else:
        self.device = torch.device("cpu")

  def init_weights(self):
    # TODO ========================
    embedding_weights = np.random.uniform(low=-0.1, high=0.1, size=(self.vocab_size, self.emb_size))
    self.embedding.weight.data.copy_(torch.from_numpy(embedding_weights))

    torch.nn.init.zeros_(self.output_layer.bias)
    torch.nn.init.uniform_(self.output_layer.weight, a=-0.1, b=0.1)


    rng = np.sqrt(1.0/self.hidden_size)
    for i in range(len(self.r_list) -1 ):
        torch.nn.init.uniform_(self.r_list[i].bias, a=-rng, b=rng)
        torch.nn.init.uniform_(self.r_list[i].weight, a=-rng, b=rng)
        torch.nn.init.uniform_(self.z_list[i].bias, a=-rng, b=rng)
        torch.nn.init.uniform_(self.z_list[i].weight, a=-rng, b=rng)
        torch.nn.init.uniform_(self.hp_list[i].bias, a=-rng, b=rng)
        torch.nn.init.uniform_(self.hp_list[i].weight, a=-rng, b=rng)

    return
    
  def init_hidden(self):
    # TODO ========================
    return torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=self.device)

  def forward(self, inputs, hidden):
    # TODO ========================

    output_list = []
    prev_hidden = hidden
    for t in range(self.seq_len):
        input_t = inputs[t]
        x_t = self.embedding(input_t)
        x_t = self.input_dropout(x_t)
        new_hidden = []
        for i in range(self.num_layers):
            concatenated_input = torch.cat([x_t, prev_hidden[i]], dim=1)

            r_t = torch.sigmoid(self.r_list[i](concatenated_input))
            z_t = torch.sigmoid(self.z_list[i](concatenated_input))
            
            new_h_tm1 = r_t * hidden[i]
            concatenated_h_input = torch.cat([x_t, new_h_tm1], dim=1)

            hp_t = torch.tanh(self.hp_list[i](concatenated_h_input))

            h_t = (1-z_t) * (hidden[i]) + (z_t * hp_t)
            h_t = self.dropout_list[i](h_t)

            x_t = h_t
            new_hidden.append(h_t)
        prev_hidden = new_hidden    
        output = self.output_layer(x_t)
        output_list.append(output)

    return torch.stack(output_list), torch.stack(prev_hidden)


  def generate(self, input, hidden, generated_seq_len):
    # TODO ========================
    return samples



# Problem 3
##############################################################################
#
# Code for the Transformer model
#
##############################################################################

"""
Implement the MultiHeadedAttention module of the transformer architecture.
All other necessary modules have already been implemented for you.

We're building a transfomer architecture for next-step prediction tasks, and 
applying it to sequential language modelling. We use a binary "mask" to specify 
which time-steps the model can use for the current prediction.
This ensures that the model only attends to previous time-steps.

The model first encodes inputs using the concatenation of a learned WordEmbedding 
and a (in our case, hard-coded) PositionalEncoding.
The word embedding maps a word's one-hot encoding into a dense real vector.
The positional encoding 'tags' each element of an input sequence with a code that 
identifies it's position (i.e. time-step).

These encodings of the inputs are then transformed repeatedly using multiple
copies of a TransformerBlock.
This block consists of an application of MultiHeadedAttention, followed by a 
standard MLP; the MLP applies *the same* mapping at every position.
Both the attention and the MLP are applied with Resnet-style skip connections, 
and layer normalization.

The complete model consists of the embeddings, the stacked transformer blocks, 
and a linear layer followed by a softmax.
"""

#This code has been modified from an open-source project, by David Krueger.
#The original license is included below:
#MIT License
#
#Copyright (c) 2018 Alexander Rush
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.



#----------------------------------------------------------------------------------

# TODO: implement this class
class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, n_units, dropout=0.1):
        """
        n_heads: the number of attention heads
        n_units: the number of input and output units
        dropout: probability of DROPPING units
        """
        super(MultiHeadedAttention, self).__init__()
        # This sets the size of the keys, values, and queries (self.d_k) to all 
        # be equal to the number of output units divided by the number of heads.
        self.d_k = n_units // n_heads
        self.sq_d_k = np.sqrt(self.d_k)
        # This requires the number of n_heads to evenly divide n_units.
        assert n_units % n_heads == 0
        self.n_heads = n_heads
        self.n_units = n_units
        self.Q_list = []
        self.V_list = []
        self.K_list = []


        for i in range(n_heads):
            self.Q_list.append(nn.Linear(self.n_units, self.d_k))
            self.V_list.append(nn.Linear(self.n_units, self.d_k))
            self.K_list.append(nn.Linear(self.n_units, self.d_k))

        self.output_layer = nn.Linear(n_units, n_units)


        self.list_of_Q_modules = nn.ModuleList(self.Q_list)
        self.list_of_V_modules = nn.ModuleList(self.V_list)
        self.list_of_K_modules = nn.ModuleList(self.K_list)

        self.init_weights()

        # TODO: create/initialize any necessary parameters or layers
        # Initialize all weights and biases uniformly in the range [-k, k],
        # where k is the square root of 1/n_units.
        # Note: the only Pytorch modules you are allowed to use are nn.Linear 
        # and nn.Dropout
        # ETA: you can also use softmax
        # ETA: you can use the "clones" function we provide.
        # ETA: you can use masked_fill

    def init_weights(self):
        k = np.sqrt(1./self.n_units)
        torch.nn.init.uniform_(self.output_layer.bias, a=-k, b=k)
        torch.nn.init.uniform_(self.output_layer.weight,a=-k, b=k)


        for i in range(len(self.Q_list)):
            torch.nn.init.uniform_(self.Q_list[i].bias, a=-k, b=k)
            torch.nn.init.uniform_(self.Q_list[i].weight, a=-k, b=k)
            torch.nn.init.uniform_(self.V_list[i].bias, a=-k, b=k)
            torch.nn.init.uniform_(self.V_list[i].weight, a=-k, b=k)
            torch.nn.init.uniform_(self.K_list[i].bias, a=-k, b=k)
            torch.nn.init.uniform_(self.K_list[i].weight, a=-k, b=k)


        
    def forward(self, query, key, value, mask=None):
        # TODO: implement the masked multi-head attention.
        # query, key, and value correspond to Q, K, and V in the latex, and 
        # they all have size: (batch_size, seq_len, self.n_units)
        # mask has size: (batch_size, seq_len, seq_len)
        # As described in the .tex, apply input masking to the softmax 
        # generating the "attention values" (i.e. A_i in the .tex)
        # Also apply dropout to the attention values.

        h_list = []
        for i in range(self.n_heads):
            new_query = self.Q_list[i](query)
            new_key = self.K_list[i](key)
            new_value = self.V_list[i](value)

            flat_query = new_query.view(new_query.shape[0] * new_query.shape[1], new_query.shape[2])
            flat_key = new_key.view(new_key.shape[0] * new_key.shape[1], new_key.shape[2])
            attention = torch.bmm(flat_query.unsqueeze(2), flat_key.unsqueeze(1))
            attention /= self.sq_d_k
            attention = attention.view(new_query.shape[0], new_query.shape[1], new_query.shape[2], new_query.shape[2])
            import ipdb
            ipdb.set_trace()
            attention = torch.softmax(attention, dim=3)

            new_attention = attention.view(attention.shape[0] * attention.shape[1], attention.shape[2], attention.shape[3])
            new_value2 = new_value.view(new_value.shape[0] * new_value.shape[1], new_value.shape[2]).unsqueeze(2)
            h_i = torch.bmm(new_attention, new_value2).squeeze(2).view(new_query.shape[0], new_query.shape[1], -1)
            h_list.append(h_i)#




        h = torch.cat(h_list, dim=2)
        flat_h = h.view(h.shape[0] * h.shape[1], h.shape[2])
        final_attention = self.output_layer(flat_h).view(new_query.shape[0], new_query.shape[1], -1)
        return final_attention
        # size: (batch_size, seq_len, self.n_units)



model = MultiHeadedAttention(10,100)


#----------------------------------------------------------------------------------
# The encodings of elements of the input sequence

class WordEmbedding(nn.Module):
    def __init__(self, n_units, vocab):
        super(WordEmbedding, self).__init__()
        self.lut = nn.Embedding(vocab, n_units)
        self.n_units = n_units

    def forward(self, x):
        #print (x)
        return self.lut(x) * math.sqrt(self.n_units)


class PositionalEncoding(nn.Module):
    def __init__(self, n_units, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_units)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, n_units, 2).float() *
                             -(math.log(10000.0) / n_units))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)



#----------------------------------------------------------------------------------
# The TransformerBlock and the full Transformer


class TransformerBlock(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerBlock, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualSkipConnectionWithLayerNorm(size, dropout), 2)
 
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) # apply the self-attention
        return self.sublayer[1](x, self.feed_forward) # apply the position-wise MLP


class TransformerStack(nn.Module):
    """
    This will be called on the TransformerBlock (above) to create a stack.
    """
    def __init__(self, layer, n_blocks): # layer will be TransformerBlock (below)
        super(TransformerStack, self).__init__()
        self.layers = clones(layer, n_blocks)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class FullTransformer(nn.Module):
    def __init__(self, transformer_stack, embedding, n_units, vocab_size):
        super(FullTransformer, self).__init__()
        self.transformer_stack = transformer_stack
        self.embedding = embedding
        self.output_layer = nn.Linear(n_units, vocab_size)
        
    def forward(self, input_sequence, mask):
        embeddings = self.embedding(input_sequence)
        return F.log_softmax(self.output_layer(self.transformer_stack(embeddings, mask)), dim=-1)


def make_model(vocab_size, n_blocks=6, 
               n_units=512, n_heads=16, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(n_heads, n_units)
    ff = MLP(n_units, dropout)
    position = PositionalEncoding(n_units, dropout)
    model = FullTransformer(
        transformer_stack=TransformerStack(TransformerBlock(n_units, c(attn), c(ff), dropout), n_blocks),
        embedding=nn.Sequential(WordEmbedding(n_units, vocab_size), c(position)),
        n_units=n_units,
        vocab_size=vocab_size
        )
    
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


#----------------------------------------------------------------------------------
# Data processing

def subsequent_mask(size):
    """ helper function for creating the masks. """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, x, pad=0):
        self.data = x
        self.mask = self.make_mask(self.data, pad)
    
    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."
        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask


#----------------------------------------------------------------------------------
# Some standard modules

class LayerNorm(nn.Module):
    "layer normalization, as in: https://arxiv.org/abs/1607.06450"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ResidualSkipConnectionWithLayerNorm(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(ResidualSkipConnectionWithLayerNorm, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class MLP(nn.Module):
    """
    This is just an MLP with 1 hidden layer
    """
    def __init__(self, n_units, dropout=0.1):
        super(MLP, self).__init__()
        self.w_1 = nn.Linear(n_units, 2048)
        self.w_2 = nn.Linear(2048, n_units)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

