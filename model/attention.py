import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AttentionHelper(nn.Module):
    def __init__(self):
        super(AttentionHelper, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def scalar_dot_att(self, proj_query, proj_key, proj_val, padding_mask):
        m, c1, l1 = proj_query.shape
        m, c2, l2 = proj_key.shape
        assert c1 == c2
        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)
        attention = energy / np.sqrt(c1)
        if padding_mask is not None:
            attention = attention + torch.log(padding_mask + 1e-6)
        attention = self.softmax(attention)
        if padding_mask is not None:
            attention = attention * padding_mask
        attention = attention.permute(0, 2, 1)
        out = torch.bmm(proj_val, attention)
        return out, attention

class AttLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type):
        super(AttLayer, self).__init__()
        self.query_conv = nn.Conv1d(in_channels=q_dim, out_channels=q_dim // r1, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=k_dim, out_channels=k_dim // r2, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=v_dim, out_channels=v_dim // r3, kernel_size=1)
        self.conv_out = nn.Conv1d(in_channels=v_dim // r3, out_channels=v_dim, kernel_size=1)
        self.bl = bl
        self.stage = stage
        self.att_type = att_type
        assert self.stage in ['encoder','decoder']
        self.att_helper = AttentionHelper()

    def forward(self, x1, x2, mask):
        query = self.query_conv(x1)
        key = self.key_conv(x1)
        if self.stage == 'decoder':
            assert x2 is not None
            value = self.value_conv(x2)
        else:
            value = self.value_conv(x1)

        if self.att_type == 'sliding_att':
            return self._sliding_window_self_att(query, key, value, mask)
        else:
            return self._normal_self_att(query, key, value, mask)

    def _normal_self_att(self, q, k, v, mask):
        m_batchsize, _, L = q.size()
        padding_mask = torch.ones((m_batchsize, 1, L)).to(device) * mask[:,0:1,:]
        output, _ = self.att_helper.scalar_dot_att(q, k, v, padding_mask)
        output = self.conv_out(F.relu(output))
        return output * mask[:, 0:1, :]

    def _sliding_window_self_att(self, q, k, v, mask):
        m_batchsize, c1, L_original = q.size()
        _, c2, _ = k.size()
        _, c3, _ = v.size()

        L_padded = L_original
        if L_original % self.bl != 0:
            pad_len = self.bl - (L_original % self.bl)
            L_padded = L_original + pad_len
            q = F.pad(q, (0, pad_len))
            k = F.pad(k, (0, pad_len))
            v = F.pad(v, (0, pad_len))

        nb = L_padded // self.bl
        q_reshaped = q.reshape(m_batchsize, c1, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c1, self.bl)

        half_bl = self.bl // 2
        window_size = self.bl + 2 * half_bl

        k_padded_window = F.pad(k, (half_bl, half_bl))
        v_padded_window = F.pad(v, (half_bl, half_bl))
        mask_padded_window = F.pad(mask, (0, L_padded - mask.size(2)))
        mask_padded_window = F.pad(mask_padded_window, (half_bl, half_bl))

        k_windows = k_padded_window.unfold(dimension=2, size=window_size, step=self.bl)
        v_windows = v_padded_window.unfold(dimension=2, size=window_size, step=self.bl)
        mask_windows = mask_padded_window.unfold(dimension=2, size=window_size, step=self.bl)

        k_reshaped = k_windows.permute(0, 2, 1, 3).reshape(m_batchsize * nb, c2, window_size)
        v_reshaped = v_windows.permute(0, 2, 1, 3).reshape(m_batchsize * nb, c3, window_size)
        mask_reshaped = mask_windows.permute(0, 2, 1, 3).reshape(m_batchsize * nb, 1, window_size)

        output, _ = self.att_helper.scalar_dot_att(q_reshaped, k_reshaped, v_reshaped, mask_reshaped)
        output = self.conv_out(F.relu(output))
        out_channels = self.conv_out.out_channels
        output = output.reshape(m_batchsize, nb, out_channels, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize, out_channels, L_padded)
        output = output[:, :, :L_original]
        return output * mask[:, 0:1, :]

class MultiHeadAttLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type, num_head):
        super(MultiHeadAttLayer, self).__init__()
        self.conv_out = nn.Conv1d(v_dim * num_head, v_dim, 1)
        self.layers = nn.ModuleList([
            AttLayer(q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type) for _ in range(num_head)
        ])
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x1, x2, mask):
        out = torch.cat([layer(x1, x2, mask) for layer in self.layers], dim=1)
        out = self.conv_out(self.dropout(out))
        return out