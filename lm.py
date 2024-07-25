import argparse
import logging
import sys

from model import CustomLSTM, LSTM, RNN

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def sample(device: torch.device,
           model: nn.Module,
           output_seq_size: int) -> list[int]:
    """
    Sample an output sequence from a model

    :param device: device to sample int
    :param model: model to sample from
    :param output_seq_size: size of the output sequence
    :returns: list of sampled indices
    """
    hidden_state = None

    # store output as list of indices
    sampled_output = []
    # create an input tensor from a random index/character in the input set
    random_idx = np.random.randint(model.input_size)
    seq = torch.tensor(random_idx).reshape(1, 1).to(device)

    for _ in range(output_seq_size):
        output, hidden_state = model(seq, hidden_state)

        # normalize output into probability distribution over all characters
        probs = F.softmax(torch.squeeze(output), dim=0)
        dist = torch.distributions.categorical.Categorical(probs)

        # sample from the distribution and append to list
        sampled_idx = dist.sample()
        sampled_output.append(sampled_idx.item())

        # reset sequence to sampled char for next loop iteration
        seq[0][0] = sampled_idx.item()
    return sampled_output


def get_model_type(model_arch: str):
    """
    Given a text name, return a model type to construct

    :param model_arch: model architecture to retrieve class for
    :raises ValueError: if model_arch is not supported
    """
    if model_arch == 'rnn':
        return RNN
    elif model_arch == 'clstm':
        return CustomLSTM
    elif model_arch == 'lstm':
        return LSTM
    else:
        raise ValueError(f'Unrecognized model arch: {model_arch}')


def train_model(logger: logging.Logger,
                device: torch.device,
                args: argparse.Namespace):
    """
    Train the model

    :param logger: logger to use for debugging info
    :param device: device to train on
    :param args: command-line args to use
    """
    # load the corpus
    logger.info(f'Training mode with parameters: {args}')
    with open(args.corpus, 'r') as f:
        corpus = f.read()
    logger.info(f'Loaded corpus {args.corpus}')

    # reduce corpus size to test model: corpus = corpus[:(2** 8)]
    unique_chars = sorted(set(corpus))
    vocab_size = len(unique_chars)
    logger.info(f'Unique characters in corpus: {vocab_size}')

    # create mappings between chars and indices
    ch_to_ix = {ch: ix for ix, ch in enumerate(unique_chars)}
    ix_to_ch = {ix: ch for ix, ch in enumerate(unique_chars)}

    # convert string corpus into Pytorch tensors
    data = [ch_to_ix[ch] for ch in corpus]
    data = torch.tensor(data).to(device)

    # reshape into tensor format: num_chars x 1
    data = torch.unsqueeze(data, dim=1)

    # create model
    logger.info(f'Using model arch: {args.model_arch}')
    model_init = get_model_type(args.model_arch)
    model = model_init(vocab_size, args.hidden_size, vocab_size, args.num_layers).to(device)

    # create loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # load a semi-trained model to continue training
    num_pretrained_epochs = 0
    if args.load_file:
        training_state_dict = torch.load(args.load_file)
        model.load_state_dict(training_state_dict['model_state_dict'])
        optimizer.load_state_dict(training_state_dict['optimizer_state_dict'])
        num_pretrained_epochs = training_state_dict['num_trained_epochs']
        logger.info(f'Loaded model {args.load_file} pre-trained for {num_pretrained_epochs} epochs')

    num_batches = len(data) // args.sequence_size
    for e in range(args.num_epochs):
        epoch_loss = 0
        hidden_state = None

        for i in range(0, len(data), args.sequence_size):
            # extract source and target sequences of len sequence_size
            source = data[i:i+args.sequence_size]
            # target sequence is offset by 1 char
            target = data[i+1:i+args.sequence_size+1]

            # stop at last uneven batch
            if source.size(0) > target.size(0):
                break

            # run source (and hidden state) through model and compute loss of target set
            output, hidden_state = model(source, hidden_state)
            loss = criterion(torch.squeeze(output), torch.squeeze(target))

            # periodically log intermediate loss
            batch_num = i // args.sequence_size
            if batch_num % args.log_interval == 0:
                logger.info(f'Batch {batch_num}/{num_batches}:\t{loss.item():.4f}')

            # compute gradients
            optimizer.zero_grad()
            loss.backward()

            # clip the gradient to prevent exploding gradient!
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            # update parameters
            optimizer.step()

            epoch_loss += loss.item() / len(source)

        logger.info(f'Epoch {e+1}:\t {epoch_loss:.4f}')

        # save model and auxiliary information to load again
        torch.save({
            'vocab_size': vocab_size,
            'hidden_size': args.hidden_size,
            'ix_to_ch': ix_to_ch,
            'num_trained_epochs': num_pretrained_epochs + e + 1,
            'num_layers': args.num_layers,
            'model_arch': args.model_arch,

            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, args.save_file)
        logger.info(f'Saved data to {args.save_file}')

        # sample output every epoch
        sampled_output = ''.join(ix_to_ch[i] for i in sample(device, model, args.output_sequence_size))
        logger.info(f'\n{sampled_output}\n')
    logger.info('Finished training!')


def eval_model(logger: logging.Logger, device: torch.device, args: argparse.Namespace):
    """
    Evaluate the model by sampling from it

    :param logger: logger to use for debugging info
    :param device: device to train on
    :param args: command-line args to use
    """
    # load model
    data = torch.load(args.load_file)
    logger.info(f"Loading model {args.load_file} pre-trained for {data['num_trained_epochs']} epochs")
    model_initializer = get_model_type(data['model_arch'])
    vocab_size = data['vocab_size']
    model = model_initializer(vocab_size, data['hidden_size'], vocab_size, data['num_layers']).to(device)
    model.load_state_dict(data['model_state_dict'])

    # sample output from model and join together characters into string
    sampled_output = ''.join(data['ix_to_ch'][i] for i in sample(device, model, args.output_sequence_size))
    logger.info(f'Sampling {args.output_sequence_size} chars...')
    logger.info(f'\n{sampled_output}\n')


def main() -> None:
    # setup a logger and logging format
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    # log to the console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # log to a file too
    file_handler = logging.FileHandler('log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # use CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device {device}')

    # define subparsers/subcommands for running this file
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    train_parser = subparsers.add_parser('train')
    # train_model(...) will be called when we invoke train subparser
    train_parser.set_defaults(func=lambda args: train_model(logger, device, args))
    train_parser.add_argument('--corpus', type=str, required=True)
    train_parser.add_argument('--hidden-size', type=int, default=512)
    train_parser.add_argument('--sequence-size', type=int, default=128)
    train_parser.add_argument('--num-epochs', type=int, default=32)
    train_parser.add_argument('--clip-grad', type=float, default=5.)
    train_parser.add_argument('--save-file', type=str, default='model.pth')
    train_parser.add_argument('--load-file', type=str)
    train_parser.add_argument('--model-arch', choices=['rnn', 'clstm', 'lstm'], default='rnn')
    train_parser.add_argument('--log-interval', type=int, default=32)
    train_parser.add_argument('--learning-rate', type=float, default=0.002)
    train_parser.add_argument('--output-sequence-size', type=int, default=128)
    train_parser.add_argument('--num-layers', type=int, default=4)

    eval_parser = subparsers.add_parser('eval')
    # eval_model(...) will be called when we invoke eval subparser
    eval_parser.set_defaults(func=lambda args: eval_model(logger, device, args))
    eval_parser.add_argument('--load-file', type=str, required=True)
    eval_parser.add_argument('--output-sequence-size', type=int, default=128)

    # invoke train_model(..., args)/eval_model(..., args)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
