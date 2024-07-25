# Recurrent Neural Networks

This repo hosts some implementations of Recurrent Neural Networks (RNNs) for educational purposes.

# Training

Train the model by running `python3 lm.py train` with a plain-text corpus

```bash
python3 lm.py train --corpus shakespeare.txt
```

There are options to change hyperparameters such as the the learning rate, model architecture, hidden state size, and more. Run `python3 lm.py train --help` to see all of the options.

# Evaluation

Evaluate the text generation quality of the model by using the `eval` subparser and providing a previously-trained Pytorch checkpoint.

```bash
python3 lm.py eval --load-file lstm-e32.pth
```
