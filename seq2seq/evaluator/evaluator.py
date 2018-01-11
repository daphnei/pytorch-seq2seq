from __future__ import print_function, division

import torch
import torchtext

import seq2seq
from seq2seq.loss import NLLLoss

class Evaluator(object):
  """ Class to evaluate models with given datasets.

  Args:
    loss (seq2seq.loss, optional): loss for evaluator (default: seq2seq.loss.NLLLoss)
    batch_size (int, optional): batch size for evaluator (default: 64)
  """

  def __init__(self, loss=NLLLoss(), batch_size=64):
    self.loss = loss
    self.batch_size = batch_size

  def evaluate(self, model, data):
    """ Evaluate a model on given dataset and return performance.

    Args:
      model (seq2seq.models): model to evaluate
      data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against

    Returns:
      loss (float): loss of the given model on the given dataset
    """
    model.eval()

    loss = self.loss
    loss.reset()
    match = 0
    total = 0

    device = None if torch.cuda.is_available() else -1
    batch_iterator = torchtext.data.BucketIterator(
      dataset=data, batch_size=self.batch_size,
      sort=True, sort_key=None,
      device=device, train=False)
    tgt_vocab = data.fields[seq2seq.field_names[0]].vocab
    pad = tgt_vocab.stoi[data.fields[seq2seq.field_names[0]].pad_token]

    num_sequences = len(seq2seq.field_names)

    for batch in batch_iterator:
      input_variables_list = []
      input_lengths_list = []
      for idx in xrange(0, num_sequences - 1):
        input_variable, input_lengths_tensor = getattr(
          batch, seq2seq.field_names[idx])

        input_variables_list.append(input_variable)
        input_lengths_list.append(input_lengths_tensor.tolist())

      target_variables, target_lengths = getattr(batch, seq2seq.field_names[num_sequences - 1])
      model_out = model(
          input_variables_list=input_variables_list,
          input_lengths_list = input_lengths_list,
          target_variable=target_variables)
      if model_out is None:
        continue
      decoder_outputs, decoder_hidden, other = model_out

      import pdb; pdb.set_trace()
      # Evaluation
      seqlist = other['sequence']
      for step, step_output in enumerate(decoder_outputs):
        target = target_variables[:, step + 1]
        loss.eval_batch(step_output.view(target_variables.size(0), -1), target)

        non_padding = target.ne(pad)
        correct = seqlist[step].view(-1).eq(target).masked_select(non_padding).sum().data[0]
        match += correct
        total += non_padding.sum().data[0]

    if total == 0:
      accuracy = float('nan')
    else:
      accuracy = match / total

    return loss.get_loss(), accuracy
