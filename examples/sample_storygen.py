import os
import argparse
import logging
import glob
import io

import torch
from torch.optim.lr_scheduler import StepLR
import torchtext

import seq2seq
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity
from seq2seq.optim import Optimizer
from seq2seq.dataset import SequenceField
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint

from torchtext.data.example import Example

try:
  raw_input      # Python 2
except NameError:
  raw_input = input  # Python 3

# Sample usage:
#   # training
#   python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH
#   # resuming from the latest checkpoint of the experiment
#    python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --resume
#    # resuming from a specific checkpoint
#    python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --load_checkpoint $CHECKPOINT_DIR

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', action='store', dest='train_path',
          help='Path to train data')
parser.add_argument('--dev_path', action='store', dest='dev_path',
          help='Path to dev data')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment',
          help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
          help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume',
          default=False,
          help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--max_len', default=60, type=int,
          help='Maximum number of characters in a sentence. (Discards all sentences longer than this.)')
parser.add_argument('--log-level', dest='log_level',
          default='info',
          help='Logging level.')

opt = parser.parse_args()

class StoryDataset(torchtext.data.Dataset):
  def __init__(self, fields, path, extension='.txt', **kwargs):
    examples = []

    num_sequences = len(fields)

    data_files = glob.glob(os.path.join(path, '*' + extension))
    for data_file in data_files:
      # Read the file line by line, and create examples from series
      # of num_sequences consecutive lines
      with io.open(os.path.expanduser(data_file), encoding="utf8") as f:
        line_buffer = []
        for line in f:
          if len(line_buffer) == num_sequences:
            # Make a new example
            example = Example.fromlist(line_buffer, fields)
            examples.append(example)

            # Remove the first sentence
            line_buffer.pop(0)
          line_buffer.append(line)

    print('Found %d examples' % (len(examples)))
    super(StoryDataset, self).__init__(examples, fields, **kwargs)

    def foo(x):
      sort_keys = []
      for i in xrange(0, len(fields)):
        example = getattr(x, fields[i][0])
        sort_keys.append(len(example))
      return sort_keys

    self.sort_key = foo #lambda x: len(x.field_0)

def prepare_dataset(num_sequences):
  # Create fields for each sentence in sentence sequence
  fields = []
  for i in xrange(num_sequences):
    fields.append(('field_' + str(i), SequenceField(include_lengths=True)))

  # Filters out any examples with too-long sentences
  def len_filter(example):
    for i in xrange(0, num_sequences):
      field_name = 'field_' + str(i)
      field = getattr(example, field_name)
      if len(field) > opt.max_len:
        return False
    return True
  
  train = StoryDataset(
      fields=fields,
      path=opt.train_path,
      filter_pred=len_filter
  )
  dev = StoryDataset( 
      fields=fields,
      path=opt.dev_path,
      filter_pred=len_filter)

  # Create a vocabulary from only the first field. Then use this vocabulary
  # for all the other fields.
  train.fields['field_0'].build_vocab(train, max_size=70000)
  for i in xrange(1, len(train.fields)):
    train.fields['field_' + str(i)].vocab = train.fields['field_0'].vocab

  return train, dev

if __name__ == '__main__':
  LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
  logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
  logging.info(opt)

  logging.info('===USING GPU %s===' % (str(torch.cuda.current_device())))

  if opt.load_checkpoint is not None:
    logging.info("loading checkpoint from {}".format(os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab
  else:
    num_sequences = 3

    train, dev = prepare_dataset(num_sequences)
    first_field = train.fields['field_0']

    input_vocab = first_field.vocab
    output_vocab = input_vocab

    seq2seq.field_names = [('field_%d' % i) for i in xrange(0, num_sequences)] 
    seq2seq.vocab = input_vocab

    # Prepare loss
    weight = torch.ones(len(output_vocab))
    pad = output_vocab.stoi[first_field.pad_token]
    loss = Perplexity(weight, pad)
    if torch.cuda.is_available():
      loss.cuda()

    batch_size = 2 #32

    seq2seq = None
    optimizer = None
    if not opt.resume:
      # Initialize model
      hidden_size = 512
      n_layers = 2
      bidirectional = True
      encoder = EncoderRNN(len(input_vocab), opt.max_len, hidden_size,
                 n_layers=n_layers, bidirectional=bidirectional, variable_lengths=True)
      decoder = DecoderRNN(len(output_vocab), opt.max_len, hidden_size * 2 if bidirectional else hidden_size,
                 dropout_p=0.2, use_attention=True, bidirectional=bidirectional,
                 n_layers=n_layers, eos_id=first_field.eos_id, sos_id=first_field.sos_id)

      seq2seq = Seq2seq(encoder, decoder, batch_size, num_sequences)
      if torch.cuda.is_available():
        seq2seq.cuda()

      for param in seq2seq.parameters():
        param.data.uniform_(-0.08, 0.08)

      # Optimizer and learning rate scheduler can be customized by
      # explicitly constructing the objects and pass to the trainer.
      #
      # optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters()), max_grad_norm=5)
      # scheduler = StepLR(optimizer.optimizer, 1)
      # optimizer.set_scheduler(scheduler)

    # train
    t = SupervisedTrainer(loss=loss, batch_size=batch_size,
              checkpoint_every=50,
              print_every=10, expt_dir=opt.expt_dir)

    seq2seq = t.train(seq2seq, train,
            num_epochs=6, dev_data=dev,
            optimizer=optimizer,
            teacher_forcing_ratio=0.5,
            resume=opt.resume)

  predictor = Predictor(seq2seq, input_vocab)

  while True:
    seq_str = raw_input("Type in a source sequence:")
    seq_1 = [first_field.SYM_SOS] + seq_str.strip().split() + [first_field.SYM_EOS]
    seq_str = raw_input("Type in a source sequence:")
    seq_2 = [first_field.SYM_SOS] + seq_str.strip().split() + [first_field.SYM_EOS]
    print(predictor.predict([seq_1, seq_2]))
