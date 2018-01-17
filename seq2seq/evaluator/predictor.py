import torch
from torch.autograd import Variable

class Predictor(object):

    def __init__(self, model, vocab): 
        """
        Predictor class to evaluate for a given model.
        Args:
            model (seq2seq.models): trained model. This can be loaded from a checkpoint
                using `seq2seq.util.checkpoint.load`
            src_vocab (seq2seq.dataset.vocabulary.Vocabulary): source sequence vocabulary
            tgt_vocab (seq2seq.dataset.vocabulary.Vocabulary): target sequence vocabulary
        """
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()
        self.vocab = vocab


    def predict(self, src_sequences):
        """ Make prediction given `src_seq` as input.

        Args:
            src_seq (list): A list of sequences. Each sequence consists of a list of tokens in source language

        Returns:
            tgt_seq (list): list of tokens in target language as predicted
            by the pre-trained model
        """
        src_id_sequences = []
        src_id_lengths = []
        for src_seq in src_sequences:  
          src_id_seq = Variable(torch.LongTensor([self.vocab.stoi[tok] for tok in src_seq]),
                                volatile=True).view(1, -1)
          if torch.cuda.is_available():
              src_id_seq = src_id_seq.cuda()

          src_id_sequences.append(src_id_seq)
          src_id_lengths.append([src_id_seq.shape[1]])

        softmax_list, _, other = self.model(src_id_sequences, src_id_lengths)
        length = other['length'][0]
        tgt_id_seq = [other['sequence'][di][0].data[0] for di in range(length)]
        tgt_seq = [self.vocab.itos[tok] for tok in tgt_id_seq]
        return tgt_seq
