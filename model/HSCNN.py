# coding: utf-8


import torch
from torchsummary import summary


class HSCNN(torch.nn.Module):

    def __init__(self, input_ch, output_ch, *args, feature=64, layer_num=9, **kwargs):
        super(HSCNN, self).__init__()
        self.activation = None
        self.output_norm = None
        self.activation = kwargs.get('activation', 'leaky')
        self.residual_shortcut = torch.nn.Identity()
        self.start_conv = torch.nn.Conv2d(input_ch, output_ch, 3, 1, 1)
        self.patch_extraction = torch.nn.Conv2d(output_ch, 64, 3, 1, 1)
        feature_map = [torch.nn.Conv2d(feature, feature, 3, 1, 1) for _ in range(layer_num - 1)]
        self.feature_map = torch.nn.Sequential(*feature_map)
        self.residual_conv = torch.nn.Conv2d(feature, output_ch, 3, 1, 1)

    def forward(self, x):

        x = self.start_conv(x)
        x_in = self.residual_shortcut(x)
        x = self._activation_fn(self.patch_extraction(x))
        for feature_map in self.feature_map:
            x = self._activation_fn(feature_map(x))
        output = self.residual_conv(x) + x_in
        return output


    def _activation_fn(self, x):
        if self.activation == 'relu':
            return torch.relu(x)
        elif self.activation == 'swish':
            return swish(x)
        elif self.activation == 'mish':
            return mish(x)
        if self.activation == 'leaky':
            return torch.nn.functional.leaky_relu(x)
        else:
            return x


if __name__ == '__main__':

    model = HSCNN(1, 31)
    summary(model, (1, 64, 64))
