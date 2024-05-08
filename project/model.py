import torch
from torch import nn
import timm
from torchvision.models.swin_transformer import swin_v2_b, Swin_V2_B_Weights, swin_v2_t, Swin_V2_T_Weights

from torchvision.ops.misc import MLP

device = 'cuda' if torch.cuda.is_available() else 'mps'

class Model(nn.Module):
    def __init__(self, num_classes, num_features, model_type='swin') -> None:
        super().__init__()
        if model_type == 'vit':
            self.model = timm.create_model('eva02_large_patch14_448.mim_m38m_ft_in22k_in1k',
                                           checkpoint_path='./eva02_large_patch14_448.mim_m38m_ft_in22k_in1k/model.safetensors')
            img_features = self.model.head.in_features
            self.model.head = nn.Identity()
        elif model_type == 'swin':
            self.model = swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1)
            img_features = self.model.head.in_features
            self.model.head = nn.Identity()
        else:
            raise NotImplementedError

        hidden_features = [2 * num_features, 4 * num_features]
        self.mlp1 = MLP(num_features,
                        hidden_features,
                        activation_layer=nn.GELU,
                        norm_layer=nn.LayerNorm,
                        inplace=None,
                        dropout=0.2)
        
        in_features = hidden_features[-1] + img_features
        hidden_features = [2 * in_features, 4 * in_features]
        self.mlp2 = MLP(in_features,
                        hidden_features,
                        activation_layer=nn.GELU,
                        norm_layer=nn.LayerNorm,
                        inplace=None,
                        dropout=0.2)
        self.out = nn.Linear(hidden_features[-1], num_classes)

        # TODO: add auxilary head.

    def get_features(self, img, features):
        img_feats = self.model(img)
        feats = self.mlp1(features)
        y = torch.concat([img_feats, feats], dim=1)
        y = self.mlp2(y)
        return y

    def forward(self, img, features):
        y = self.get_features(img, features)
        y = self.out(y)

        return y


if __name__ == "__main__":
    model = Model(6, 10, model_type='vit')
    print(model)