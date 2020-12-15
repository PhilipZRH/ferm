import torch
import torch.nn as nn
import torch.nn.functional as F


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


# for 84 x 84 inputs
OUT_DIM = {2: 39, 4: 35, 6: 31}
# for 64 x 64 inputs
OUT_DIM_64 = {2: 29, 4: 25, 6: 21}


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, output_logits=False, two_conv=False):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.dual_cam = obs_shape[0] == 6
        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.two_conv = two_conv
        if not self.two_conv:
            self.convs = nn.ModuleList(
                [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
            )
            for i in range(num_layers - 1):
                self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        else:
            self.convs1 = nn.ModuleList([nn.Conv2d(obs_shape[0] // 2, num_filters, 3, stride=2)])
            self.convs2 = nn.ModuleList([nn.Conv2d(obs_shape[0] // 2, num_filters, 3, stride=2)])
            for i in range(num_layers - 1):
                self.convs1.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
                self.convs2.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        x = torch.randn([32] + list(obs_shape))
        self.outputs = dict()

        out_dim = self.forward_conv(x, flatten=False).shape[-1]
        # out_dim = OUT_DIM_64[num_layers] if obs_shape[-1] == 64 else OUT_DIM[num_layers]
        if self.two_conv:
            self.fc = nn.Linear(2 * num_filters * out_dim * out_dim, self.feature_dim)
        else:
            self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.output_logits = output_logits

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs, flatten=True):

        if obs.max() > 1.:
            obs = obs / 255.
        self.outputs['obs'] = obs

        # if self.dual_cam:
        #     cam1 = torch.cat([obs[:, :3, :, :], torch.zeros_like(obs[:, 3:, :, :])], 1)
        #     cam2 = torch.cat([torch.zeros_like(obs[:, :3, :, :]), obs[:, 3:, :, :]], 1)
        #
        #     cam1conv = torch.relu(self.convs[0](cam1))
        #     cam2conv = torch.relu(self.convs[0](cam2))
        #     # self.outputs['cam1conv1'] = cam1conv
        #     # self.outputs['cam2conv1'] = cam2conv

        if not self.two_conv:
            conv = torch.relu(self.convs[0](obs))
            self.outputs['conv1'] = conv

            for i in range(1, self.num_layers):
                conv = torch.relu(self.convs[i](conv))
                self.outputs['conv%s' % (i + 1)] = conv
                # if self.dual_cam:
                #     cam1conv = torch.relu(self.convs[i](cam1conv))
                #     cam2conv = torch.relu(self.convs[i](cam2conv))
                #     if i == self.num_layers - 1:
                #         self.outputs['cam1conv%s' % (i + 1)] = cam1conv
                #         self.outputs['cam2conv%s' % (i + 1)] = cam2conv
        else:
            img1, img2 = torch.split(obs, [3, 3], dim=1)
            conv1 = torch.relu(self.convs1[0](img1))
            conv2 = torch.relu(self.convs2[0](img2))
            self.outputs['conv1_1'] = conv1
            self.outputs['conv2_1'] = conv2
            for i in range(1, self.num_layers):
                conv1 = torch.relu(self.convs1[i](conv1))
                conv2 = torch.relu(self.convs2[i](conv2))
                self.outputs['conv1_%s' % (i + 1)] = conv1
                self.outputs['conv2_%s' % (i + 1)] = conv2
            conv = torch.cat([conv1, conv2], dim=1)

        if flatten:
            conv = conv.reshape(conv.size(0), -1)
        return conv

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm)
            self.outputs['tanh'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        if not self.two_conv:
            for i in range(self.num_layers):
                tie_weights(src=source.convs[i], trg=self.convs[i])
        else:
            for i in range(self.num_layers):
                tie_weights(src=source.convs1[i], trg=self.convs1[i])
                tie_weights(src=source.convs2[i], trg=self.convs2[i])

    def log(self, L, step, log_networks, log_freq):

        if step % log_freq != 0 and not log_networks:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            if not self.two_conv:
                L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
            else:
                L.log_param('train_encoder/conv1_%s' % (i + 1), self.convs1[i], step)
                L.log_param('train_encoder/conv2_%s' % (i + 1), self.convs2[i], step)
        # attention = self.outputs['conv3']
        # attention = torch.abs(attention[0]).mean(axis=0)
        # attention_dim = attention.shape[0]
        # attention = attention.reshape(-1)
        # attention = F.softmax(attention)
        # attention = attention.reshape(attention_dim, attention_dim)
        # L.log_image('train_encoder/attention_img', attention, step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters, *args, two_conv=False):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_networks, log_freq):
        pass


_AVAILABLE_ENCODERS = {'pixel': PixelEncoder, 'identity': IdentityEncoder}


def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters, output_logits=False, two_conv=False
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters, output_logits, two_conv=two_conv
    )
