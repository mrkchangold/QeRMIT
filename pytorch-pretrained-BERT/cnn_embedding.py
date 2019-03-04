from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np

class CNN(nn.Module):
    def __init__(self, e_word = 256, e_char = 50, k = 5):
        super(CNN, self).__init__()
        self.e_word = e_word
        self.e_char = e_char
        self.xconv = None
        self.k = k
        self.convLayer = nn.Conv1d(in_channels = e_char, out_channels = e_word, kernel_size = k, stride=1, padding=0, bias=True)

    def forward(self, xreshaped: torch.Tensor):
        self.xconv = self.convLayer(xreshaped)
        xconv_out = nn.MaxPool1d(kernel_size = self.xconv.size()[-1] - self.k + 1)(nn.ReLU()(self.xconv)) # potentially ask question
        return xconv_out

    # def __init__(self, e_T = 50, k = 5):
    #     # e_T is transformer size
    #     #
    #     super(CNN, self).__init__()
    #     self.e_T = e_T
    #     self.qconv = None
    #     self.k = k
    #     self.convLayer = nn.Conv1d(in_channels = e_T, out_channels = e_T, kernel_size = k, stride=1, padding=0, bias=True)

    # def forward(self, qreshaped: torch.Tensor):
    #     self.qconv = self.convLayer(qreshaped)
    #     qconv_out = nn.MaxPool1d(kernel_size = self.qconv.size()[-1] - self.k + 1)(nn.ReLU()(self.qconv)) # potentially ask question
    #     return qconv_out

class Highway(nn.Module):
    def __init__(self, e_word = 256):
        super(Highway, self).__init__()
        self.projLayer = nn.Linear(in_features = e_word, out_features = e_word, bias = True)
        self.gateLayer = nn.Linear(in_features = e_word, out_features = e_word, bias = True)

    def forward(self, xconvout: torch.Tensor):
        xproj = nn.ReLU()(self.projLayer(xconvout))
        xgate = nn.Sigmoid()(self.gateLayer(xconvout))
        xhighway = torch.mul(xgate,xproj) + torch.mul(1-xgate,xconvout)
        return xhighway

class CNNEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab = None):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(CNNEmbeddings, self).__init__()

        self.embed_size = embed_size
        self.vocab = vocab
        self.e_char = 50 # given in instruction
        self.dropout = 0.3 # given in instruction

        self.cnn = CNN(e_word = embed_size, e_char = self.e_char, k = 5)
        self.hwy = Highway(e_word = embed_size)
        self.dropout = nn.Dropout(self.dropout)
        # self.embeddings = nn.Embedding(num_embeddings = len(vocab.char2id), embedding_dim = self.e_char, padding_idx = vocab.char2id['<pad>'])

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary
        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        output = self.embeddings(input)
        sentence_length, batch_size, max_word_length, e_char = output.size()
        output = output.view(-1,max_word_length,e_char)
        output = output.permute(0,2,1) #(sentxbatch) x e_char x word_len
        output = self.cnn(output) #input: (sentxbatch) x e_char x word_len
        output = output.permute(0,2,1) #input: (sentxbatch) x e_word x word_len
        output = self.hwy(output) #input (sentxbatch) x word_len x e_word
        output = output.view(sentence_length, batch_size, -1)
        return output

class QEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(QEmbeddings, self).__init__()

        self.embed_size = embed_size
        self.dropout = 0.3

        self.cnn = CNN(e_word = embed_size, e_char = embed_size, k = 5)
        self.hwy = Highway(e_word = embed_size)
        self.dropout = nn.Dropout(self.dropout)
        # self.embeddings = nn.Embedding(num_embeddings = len(vocab.char2id), embedding_dim = self.e_char, padding_idx = vocab.char2id['<pad>'])

    def forward(self, input):
        """
        """
        # print("INSIDE QEMBEDDING")
        # batch, max_q_length, e_hidden = input.size()
        output = input.permute(0,2,1).contiguous() #batch x e_hidden x max_q_length
        # print(output.grad) # false...
        output = self.cnn(output) #input: batch x e_hidden x max_q_length
        # print(output.grad) # true
        output = output.permute(0,2,1) #input: batch x e_hidden x max_q_length
        # print(output.grad) # false
        output = self.hwy(output) #input: batch x max_q_length x e_hidden
        # print(output.grad)
        # output = output.view(sentence_length, batch_size, -1) # This seems unnecessary
        # print("OUTPUT SHAPE")
        # print(output.size()) #assuming batch x 1 x e_hidden
        return output
