import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from utils.comp_dynamic_feature import comp_dynamic_feature
class Model(nn.Module):
    def __init__(self, config , device):
        super(Model, self).__init__()
        self.config = config
        self.blstm_output_size = 2 * self.config.rnn_size
        #"--tflstm_size unit size for grid lstm, 64"
        #'--tffeature_size'input size for the frequency dimension of grid lstm layer, 29
        #'--tffrequency_skip'shift of the input for the frequency dimension of grid lstm layer, 10
        #'--tflstm_layers'number of grid lstm layers, 1
        self.device=device

        self.tflstm_output_size = 2 * self.config.tflstm_size * int((self.config.input_size - self.config.tffeature_size) / self.config.tffrequency_skip + 1)
        
        # feed-forward layer
        self.forward1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.config.input_size, self.config.rnn_size),
            nn.Tanh(),
            nn.Unflatten(0,(config.batch_size,-1))
        )
        if self.config.dense_layer.lower() == 'false':
            self.forward1 = nn.Identity()
        
        # grid lstm layer
        if self.config.tflstm_size > 0:
            self.tflstm = nn.ModuleList([nn.LSTMCell(input_size=self.config.rnn_size, hidden_size=self.config.tflstm_size) for _ in range(self.config.tflstm_layers)])
            self.tflstm_output = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.tflstm_output_size, self.config.rnn_size),
                nn.Unflatten(1, (self.config.batch_size, -1, self.config.rnn_size))
            )
        
        # BLSTM layer
        self.blstm = nn.LSTM(input_size=self.config.rnn_size, hidden_size=self.config.rnn_size, num_layers=self.config.rnn_num_layers, bidirectional=True)
        self.dropout = nn.Dropout(p=1-self.config.keep_prob)
        
        # Mask estimation layer
        self.mask1 = nn.Sequential(
            nn.Linear(self.blstm_output_size, self.config.output_size),
            nn.ReLU() if self.config.mask_type.lower() == 'relu' else nn.Sigmoid()
        )
        self.mask2 = nn.Sequential(
            nn.Linear(self.blstm_output_size, self.config.output_size),
            nn.ReLU() if self.config.mask_type.lower() == 'relu' else nn.Sigmoid()
        )
    
    def forward(self, inputs, lengths,labels1,labels2):
        self.lengths=lengths
        self.labels1=nn.utils.rnn.pad_sequence(labels1,batch_first=True,padding_value=0).to(self.device)
        self.labels2=nn.utils.rnn.pad_sequence(labels2,batch_first=True,padding_value=0).to(self.device)
        # feed-forward layer
        inputs = nn.utils.rnn.pad_sequence(inputs,batch_first=True,padding_value=0).to(self.device)
        outputs=inputs.view(-1,inputs.shape[-1])
        outputs = self.forward1(outputs.float())

        # grid lstm layer
        if self.config.tflstm_size > 0:
            for layer in self.tflstm:
                outputs = layer(outputs)
            outputs = self.tflstm_output(outputs.view(-1, self.tflstm_output_size))
        
        # BLSTM layer
        outputs = nn.utils.rnn.pack_padded_sequence(outputs, lengths, batch_first=True, enforce_sorted=False)
        outputs, (fw_final_states, bw_final_states) = self.blstm(outputs)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        outputs = self.dropout(outputs)
        
        # Mask estimation layer
        mask1 = self.mask1(outputs.reshape(-1, self.blstm_output_size))
        mask2 = self.mask2(outputs.reshape(-1, self.blstm_output_size))
        mask1 = mask1.reshape(self.config.batch_size, -1, self.config.output_size)
        mask2 = mask2.reshape(self.config.batch_size, -1, self.config.output_size)
        
        self.sep1 = mask1 * inputs
        self.sep2 = mask2 * inputs
        
        return self.sep1, self.sep2
    def get_tPSA(self, labels):
        return torch.min(torch.max(labels, torch.tensor(0, dtype=labels.dtype)), self._mixed)
    def cal_loss(self):
        
        if self.config.mag_factor > 0.0:
            cost1 = torch.sum(torch.sum(torch.abs(torch.pow(self.sep1 - self.labels1, self.config.power_num)), 1) +
                        torch.sum(torch.abs(torch.pow(self.sep2 - self.labels2, self.config.power_num)), 1), 1)
            cost2 = torch.sum(torch.sum(torch.abs(torch.pow(self.sep2 - self.labels1, self.config.power_num)), 1) +
                        torch.sum(torch.abs(torch.pow(self.sep1 - self.labels2, self.config.power_num)), 1), 1)
            cost1 = self.config.mag_factor * cost1
            cost2 = self.config.mag_factor * cost2
        else:
            cost1 = torch.tensor(0.0)
            cost2 = torch.tensor(0.0)

        if self.config.del_factor > 0.0:
            sep_delta1 = comp_dynamic_feature(self.sep1, self.config.dynamic_win, self.config.batch_size, self.lengths)
            sep_delta2 = comp_dynamic_feature(self.sep2, self.config.dynamic_win, self.config.batch_size, self.lengths)
            labels_delta1 = comp_dynamic_feature(self.labels1, self.config.dynamic_win, self.config.batch_size, self.lengths)
            labels_delta2 = comp_dynamic_feature(self.labels2, self.config.dynamic_win, self.config.batch_size, self.lengths)
            cost_del1 = torch.sum(torch.sum(torch.abs(torch.pow(sep_delta1 - labels_delta1, self.config.power_num)), 1) +
                            torch.sum(torch.abs(torch.pow(sep_delta2 - labels_delta2, self.config.power_num)), 1), 1)
            cost_del2 = torch.sum(torch.sum(torch.abs(torch.pow(sep_delta2 - labels_delta1, self.config.power_num)), 1) +
                            torch.sum(torch.abs(torch.pow(sep_delta1 - labels_delta2, self.config.power_num)), 1), 1)

            cost1 += self.config.del_factor * cost_del1
            cost2 += self.config.del_factor * cost_del2

        if self.config.acc_factor > 0.0:
            sep_acc1 = comp_dynamic_feature(sep_delta1, self.config.dynamic_win, self.config.batch_size, self.lengths)
            sep_acc2 = comp_dynamic_feature(sep_delta2, self.config.dynamic_win, self.config.batch_size, self.lengths)
            labels_acc1 = comp_dynamic_feature(labels_delta1, self.config.dynamic_win, self.config.batch_size, self.lengths)
            labels_acc2 = comp_dynamic_feature(labels_delta2, self.config.dynamic_win, self.config.batch_size, self.lengths)
            cost_acc1 = torch.sum(torch.sum(torch.abs(torch.pow(sep_acc1 - labels_acc1, self.config.power_num)), 1) +
                            torch.sum(torch.abs(torch.pow(sep_acc2 - labels_acc2, self.config.power_num)), 1), 1)
            cost_acc2 = torch.sum(torch.sum(torch.abs(torch.pow(sep_acc2 - labels_acc1, self.config.power_num)), 1) +
                            torch.sum(torch.abs(torch.pow(sep_acc1 - labels_acc2, self.config.power_num)), 1), 1)

            cost1 += self.config.acc_factor * cost_acc1
            cost2 += self.config.acc_factor * cost_acc2
        cost1.to(self.device)
        cost2.to(self.device)
        cost1 = cost1 / self.lengths.float().to(self.device)
        cost2 = cost2 / self.lengths.float().to(self.device)

        # find the optimal permuration and obtain the minimum loss
        idx = (cost1 > cost2).float()
        self.loss = torch.sum(idx * cost2 + (1 - idx) * cost1)
        return self.loss
       
        