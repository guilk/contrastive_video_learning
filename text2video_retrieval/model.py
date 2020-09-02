import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import *
#from torch.autograd import Variable

# B: batch size
# N: max_sent_len
# H: hidden dim
# R: ROI
class QAN(torch.nn.Module):
    def __init__(self, params, w2v=None):
        super(QAN, self).__init__()
        self.params=params
        self.num_roi = params.num_roi
        self.hidden_dim = params.hidden_dim
        if w2v is not None:
            print('use glove')
            self.w2v = nn.Embedding.from_pretrained(w2v, freeze=False)
        else:
            self.w2v = nn.Embedding(params.vocab_size, params.tfeat_dim)
        self.step = params.step
        if params.text_encoder == 'rnn':
            self.text_encoder = rnn_encoder(emb_dim=params.tfeat_dim, hidden_dim=params.hidden_dim)
        elif params.text_encoder == 'cnn':
            self.text_encoder = cnn_encoder(emb_dim=params.tfeat_dim, hidden_dim=params.hidden_dim)
        elif params.text_encoder == 'linear':
            self.text_encoder = linear_encoder(emb_dim=params.tfeat_dim, hidden_dim=2*params.hidden_dim)
        else:
            assert(0)

        if params.vis_encoder == 'rnn':
            self.vis_encoder = rnn_encoder(emb_dim=params.vfeat_dim, hidden_dim=params.hidden_dim)
        elif params.vis_encoder == 'cnn':
            self.vis_encoder = cnn_encoder(emb_dim=params.vfeat_dim, hidden_dim=params.hidden_dim)
        elif params.vis_encoder == 'linear':
            self.vis_encoder = linear_encoder(emb_dim=params.vfeat_dim, hidden_dim=2*params.hidden_dim)
        else:
            assert(0)

        self.t_intra_attn = nn.ModuleList([attn(hidden_dim=2*self.hidden_dim) for i in range(self.step)])
        self.v_intra_attn = nn.ModuleList([attn(hidden_dim=2*self.hidden_dim) for i in range(self.step)])
        self.t_attns = nn.ModuleList([mem_attn(hidden_dim=2*self.hidden_dim) for i in range(self.step)])
        self.v_attns = nn.ModuleList([mem_attn(hidden_dim=2*self.hidden_dim) for i in range(self.step)])
        self.dropout = nn.Dropout(p=params.dp)
       
        if params.cnn2:
            self.cnn2t = nn.ModuleList([cnn_encoder(emb_dim=2*params.hidden_dim, hidden_dim=params.hidden_dim) for i in range(self.step)])
            self.cnn2v = nn.ModuleList([cnn_encoder(emb_dim=2*params.hidden_dim, hidden_dim=params.hidden_dim) for i in range(self.step)])
        self.text_mapping = MFC(params, [self.step*4*params.hidden_dim, params.hidden_dim], dropout=params.dp, have_bn=True, have_last_bn=True)
        self.vis_mapping  = MFC(params, [self.step*4*params.hidden_dim, params.hidden_dim], dropout=params.dp, have_bn=True, have_last_bn=True)
    
    def forward(self, input_caption, cmask, input_image, vmask):
        # Encoding cap
        input_caption = self.w2v(input_caption)
        e_t = self.text_encoder(input_caption, cmask)
        max_seq_len_in_batch = e_t.size(1)
        cmask = cmask[:, :max_seq_len_in_batch]
        if self.params.text_norm:
            e_t = l2norm(e_t)
        e_t = self.dropout(e_t)

        # Encoding vfeat
        if self.params.vin_norm:
            input_image = l2norm(input_image)
        e_v = self.vis_encoder(input_image, vmask)
        max_seq_len_in_batch = e_v.size(1)
        vmask = vmask[:, :max_seq_len_in_batch]
        if self.params.vis_norm:
            e_v = l2norm(e_v)
        e_v = self.dropout(e_v)
        
        for i in range(self.step):
            vm = self.v_intra_attn[i](e_v, vmask)
            tm = self.t_intra_attn[i](e_t, cmask)
            v1 = self.v_attns[i](vm, e_v, vmask)
            t1 = self.t_attns[i](tm, e_t, cmask)
            v1 = self.dropout(v1)
            t1 = self.dropout(t1)

            ct = self.cnn2t[i](e_t, cmask)
            cv = self.cnn2v[i](e_v, vmask)
            ct = self.dropout(ct)
            cv = self.dropout(cv)
            if i == 0:
                c_v = torch.cat((v1,cv), dim=1)
                c_t = torch.cat((t1,ct), dim=1)
            else:
                c_v = torch.cat((c_v, v1, cv), dim=1)
                c_t = torch.cat((c_t, t1, ct), dim=1)

        c_t = self.text_mapping(c_t)
        c_v = self.vis_mapping(c_v)
        
        c_t = l2norm(c_t, dim=-1)
        c_v = l2norm(c_v, dim=-1)
        return c_t, c_v, None, None

very_neg = -float(1e9)
def new_parameter(*size):
    out = nn.Parameter(torch.FloatTensor(*size))
    torch.nn.init.xavier_normal_(out)
    return out

class attn(nn.Module):
    def __init__(self, hidden_dim):
        super(attn, self).__init__()
        self.context_vector = new_parameter(hidden_dim,1)

    def forward(self, input, mask):
        score = torch.matmul(input, self.context_vector).squeeze() # B x R
        score = score.masked_fill_(~mask, very_neg)   # B x R
        attn_weight = F.softmax(score, dim=1).unsqueeze(2) # B x R x 1
        scored_input = input*attn_weight # B x R x H
        attended_input = torch.sum(scored_input, dim=1) # B x H
        return attended_input

class mem_attn(nn.Module):
    def __init__(self, hidden_dim):
        super(mem_attn, self).__init__()
        self.Wv = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.Wm = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.tanh = nn.Tanh()
        self.hidden_dim = hidden_dim
        self.Wv.apply(weights_init)
        self.Wm.apply(weights_init)

    def forward(self, mem, input, mask):
        bs = input.size()[0]
        input2 = self.tanh(self.Wv(input)) # B x R x H
        mem2 = self.tanh(self.Wm(mem))
        score = torch.bmm(input2, mem2.unsqueeze(2)).squeeze() # B x R
        score = score.masked_fill_(~mask, very_neg)   # B x R
        attn_weight = F.softmax(score,dim=-1).unsqueeze(2) # B x R x 1
        scored_input = input*attn_weight # B x R x H
        attned = torch.sum(scored_input, dim=1) # B x H
        return attned

class linear_encoder(torch.nn.Module):
    def __init__(self, emb_dim, hidden_dim):
        super(linear_encoder, self).__init__()
        self.encoder = nn.Linear(in_features=emb_dim, out_features=hidden_dim)
        self.tanh = nn.Tanh()
        self.encoder.apply(weights_init)
      
    def forward(self, vfeat, vamsk, tanh=False):
        output = self.encoder(vfeat)
        #output = self.tanh(output)
        return output

class cnn_encoder(torch.nn.Module):
    def __init__(self, hidden_dim, emb_dim):
        super(cnn_encoder, self).__init__()
        kernel_sizes = [2,3,4,5]
        kernel_num = int(hidden_dim/2)
        self.convs1 = nn.ModuleList([
            nn.Conv2d(1, kernel_num, (kernel_size, emb_dim), padding=(kernel_size - 1, 0))
            for kernel_size in kernel_sizes])

    def forward(self, embeds, mask):
        #print('embeds', embeds.size()) # 128,48,1024
        con_out = [F.relu(conv(embeds.unsqueeze(1))).squeeze(3) for conv in self.convs1]
        con_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in con_out]
        con_out = torch.cat(con_out, 1)

        '''
        #print('con_out', con_out[0].size()) # 128,512,49
        #length = min([x.size()[2] for x in con_out])
        length = embeds.size()[1]
        con_out = [x[:,:,1:length+1] for x in con_out]
        #print('con_out_min_length', con_out[0].size()) # 128, 512
        #con_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in con_out]
        #print('con_out_max', con_out[0].size()) # 128, 512
        #con_out = torch.cat(con_out, 1)
        #print('con_out', con_out.size()) # 128, 1536
        con_out = torch.cat(con_out, 1)
        con_out = con_out.transpose(1,2)
        #con_out = self.fc1(con_out.transpose(1,2))
        '''
        return con_out

class rnn_encoder(torch.nn.Module):
    def __init__(self, emb_dim, hidden_dim):
        super(rnn_encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(emb_dim, self.hidden_dim, bidirectional=True, batch_first=True)
        self.gru.apply(weights_init)

    def forward(self, embeds, cmask, tanh=False):
        # pack sequence
        lengths = torch.sum(cmask, dim=1).cuda()
        lengths_new, ind = torch.sort(lengths, descending=True)
        embeds = embeds[ind]
        _, inv_ind = torch.sort(ind)
        packed = pack_padded_sequence(embeds, lengths_new, batch_first=True)
        # forward
        out, _ = self.gru(packed)
        # Reshape *final* output to (batch_size, hidden_size)
        h, _ = pad_packed_sequence(out, batch_first=True)
        # bi-rnn
        h_forward = h[:,:,0:self.hidden_dim]
        h_backward = h[:,:,self.hidden_dim:]
        h2 = torch.cat((h_forward,h_backward),dim=2)
        # recover index
        h2 = h2[inv_ind]
        return h2

def xavier_init_fc(fc):
    """Xavier initialization for the fully connected layer
    """
    r = np.sqrt(6.) / np.sqrt(fc.in_features +
                             fc.out_features)
    fc.weight.data.uniform_(-r, r)
    fc.bias.data.fill_(0)

class MFC(nn.Module):
    """
    Multi Fully Connected Layers
    """
    def __init__(self, params, fc_layers, dropout, have_dp=True, have_bn=False, have_last_bn=False):
        super(MFC, self).__init__()
        self.params = params
        # fc layers
        self.n_fc = len(fc_layers)
        if self.n_fc > 1:
            if self.n_fc > 1:
                self.fc1 = nn.Linear(fc_layers[0], fc_layers[1])

            # dropout
            self.have_dp = have_dp
            if self.have_dp:
                self.dropout = nn.Dropout(p=dropout)

            # batch normalization
            self.have_bn = have_bn
            self.have_last_bn = have_last_bn
            if self.have_bn:
                if self.n_fc == 2 and self.have_last_bn:
                    self.bn_1 = nn.BatchNorm1d(fc_layers[1])

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        if self.n_fc > 1:
            xavier_init_fc(self.fc1)

    def forward(self, inputs):

        if self.n_fc <= 1:
            features = inputs

        elif self.n_fc == 2:
            features = self.fc1(inputs)
            # batch noarmalization
            if self.have_bn and self.have_last_bn:
                features = self.bn_1(features)
            if self.have_dp:
                features = self.dropout(features)

        return features
