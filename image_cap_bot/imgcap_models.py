import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_EMBED_SIZE = 2048
IMG_EMBED_BOTTLENECK = 120
WORD_EMBED_SIZE = 100
LSTM_UNITS = 300
LOGIT_BOTTLENECK = 120
PAD_IDX = 1
START = 2
END = 0
VOCAB_SIZE = 8769

with open('image_cap_bot/data/vocab.pickle', 'rb') as handle:
    vocab_inverse = pickle.load(handle)


class VeryModel(nn.Module):
    def __init__(self, lr_scheduler=None, lr_scheduler_type=None):
        super().__init__()
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_type = lr_scheduler_type
        if lr_scheduler_type not in [None, 'per_batch', 'per_epoch']:
            raise ValueError("lr_scheduler_type must be one of: None, 'per_batch', 'per_epoch'. "
                             f"Not: {lr_scheduler_type}")
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()

        self.dense1 = nn.Linear(IMG_EMBED_SIZE, IMG_EMBED_BOTTLENECK)
        self.dense2 = nn.Linear(IMG_EMBED_BOTTLENECK, LSTM_UNITS)
        self.embed_layer = nn.Embedding(VOCAB_SIZE, WORD_EMBED_SIZE)
        self.lstm1 = nn.LSTMCell(WORD_EMBED_SIZE, LSTM_UNITS)
        self.dense3 = nn.Linear(LSTM_UNITS, LOGIT_BOTTLENECK)
        self.dense4 = nn.Linear(LOGIT_BOTTLENECK, VOCAB_SIZE)

    def forward(self, batch):
        out1 = F.elu(self.dense1(batch['image_embed']))
        out2 = F.elu(self.dense2(out1))

        sentences = batch['caption'][0:batch['caption'].shape[0],
                    0:batch['caption'].shape[1] - 1].to(torch.int32)
        embed_out = self.embed_layer(sentences)

        output = []
        for i in range(sentences.shape[1]):
            hx, _ = self.lstm1(embed_out[:, i, ...], (out2, out2))
            output.append(hx)

        output_lstm = torch.stack(output, dim=1)  # .permute(1, 0, 2)
        flat_hidden_states = torch.reshape(output_lstm, [-1, LSTM_UNITS]).to(self.device)

        out3 = F.elu(self.dense3(flat_hidden_states))
        out4 = self.dense4(out3)

        flat_ground_truth = torch.reshape(batch['caption'][:, 1:], [-1]).type(torch.int64)
        flat_loss_mask = torch.not_equal(flat_ground_truth, 1)
        flat_ground_truth = flat_ground_truth * flat_loss_mask

        return flat_ground_truth, out4

    def compute_all(self, batch):
        flat_ground_truth, flat_token_logits = self.forward(batch)
        loss = F.cross_entropy(flat_token_logits, flat_ground_truth)
        # loss = F.nll_loss(F.softmax(flat_token_logits), flat_ground_truth)

        return loss

    def predict(self, image_embed, t=1, max_len=20, sample=False):
        out1 = F.elu(self.dense1(image_embed))
        out2 = F.elu(self.dense2(out1))

        caption = [START]
        embed_out = self.embed_layer(torch.Tensor((np.array([caption[-1]]))).type(torch.int32).to(self.device))
        lstm_input = embed_out
        lstm_init = (out2[None, ...], out2[None, ...])

        for _ in range(max_len):
            new_h, new_c = self.lstm1(lstm_input, lstm_init)
            next_word_probs = F.softmax(self.dense4(F.elu(self.dense3(new_h)))).ravel()
            next_word_probs = next_word_probs.detach().cpu().numpy()
            # apply temperature
            next_word_probs = next_word_probs ** (1 / t) / np.sum(next_word_probs ** (1 / t))

            if sample:
                next_word = np.random.choice(range(VOCAB_SIZE), p=next_word_probs)
            else:
                next_word = np.argmax(next_word_probs)

            caption.append(next_word)
            embed_out = self.embed_layer(torch.Tensor((np.array([caption[-1]]))).type(torch.int32).to(self.device))
            lstm_input = embed_out
            lstm_init = (new_h, new_c)

            if next_word == 0:
                break

        return list(map(vocab_inverse.get, caption))
