"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax

from pytorch_pretrained_bert.tokenization import (BasicTokenizer,
                                                  BertTokenizer,
                                                  whitespace_tokenize)
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering, BertConfig, WEIGHTS_NAME, CONFIG_NAME, BertModel

class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        # self.embed = nn.Embedding.from_pretrained(word_vectors) 
        # This is actually BERT
        self.embed = BertForQuestionAnswering.from_pretrained('bert-large-uncased')

        for name, param in self.embed.named_parameters():
            param.requires_grad = False

        self.embed_char = CNNEmbeddings(char_vectors = char_vectors, embed_size = 64) # added_flag
        # self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.proj = nn.Linear(64+1024, hidden_size, bias=False) # added_flag hardcoded
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x, xc): # added_flag xc
        # emb = self.embed(x)   # (batch_size, seq_len, embed_size)
        emb, _ = self.embed.bert(x, output_all_encoded_layers=False)

        emb_char = self.embed_char(xc) # added_flag
        emb = torch.cat((emb, emb_char), dim=2) # added_flag
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x

# good commit
class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths, batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2


########################################## added_flag
class CNN(nn.Module):
    def __init__(self, e_word = 256, e_char = 50, k = 5):
        super(CNN, self).__init__()
        self.e_word = e_word
        self.e_char = e_char
        self.xconv = None
        self.k = k
        self.convLayer = nn.Conv1d(in_channels = e_char, out_channels = e_word, kernel_size = k, stride=1, padding=2, bias=True) # added_flag padding = 2

    def forward(self, xreshaped: torch.Tensor):
        self.xconv = self.convLayer(xreshaped)
        xconv_out = nn.MaxPool1d(kernel_size = self.xconv.size()[-1])(nn.ReLU()(self.xconv)) # potentially ask question
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
    def __init__(self, char_vectors, embed_size, vocab = None):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(CNNEmbeddings, self).__init__()

        self.embed_size = 64 # hard_coded
        self.vocab = vocab
        self.e_char = 64 # hard coded
        # self.e_char = 50 # given in instruction

        self.dropout = 0.3 # given in instruction

        self.cnn = CNN(e_word = embed_size, e_char = self.e_char, k = 5)
        self.hwy = Highway(e_word = embed_size)
        self.dropout = nn.Dropout(self.dropout)
        self.embeddings = nn.Embedding.from_pretrained(char_vectors, freeze = False) # added_flag
        # self.embeddings = nn.Embedding(num_embeddings = len(vocab.char2id), embedding_dim = self.e_char, padding_idx = vocab.char2id['<pad>'])

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary
        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        # print(input.size()) #[2, 246]
        output = self.embeddings(input)
        # print(output.size()) #[2, 246, 64]
        sentence_length, batch_size, max_word_length, e_char = output.size()
        # sentence_length, batch_size, max_word_length, e_char = output.size()
        output = output.view(-1,max_word_length,e_char)
        output = output.permute(0,2,1) #(sentxbatch) x e_char x word_len
        output = self.cnn(output) #input: (sentxbatch) x e_char x word_len
        output = output.permute(0,2,1) #input: (sentxbatch) x e_word x word_len
        output = self.hwy(output) #input (sentxbatch) x word_len x e_word
        output = output.view(sentence_length, batch_size, -1)
        return output