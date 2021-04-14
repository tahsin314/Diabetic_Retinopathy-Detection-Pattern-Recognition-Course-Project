from torch import nn
import timm

class ViT(nn.Module):
    def __init__(self, pretrained_model='vit_base_patch16_384', num_class=5):
        super().__init__()
        self.backbone = timm.create_model(pretrained_model, pretrained=True)
        blocks = []
        for b in self.backbone.blocks:
            blocks.append(nn.Sequential(b))
        self.blocks = nn.Sequential(*blocks)
        # print(self.blocks[-1][0].mlp.fc2.out_features)
        # print(self.backbone.norm)
        self.in_features = self.backbone.norm.normalized_shape[0]
        self.in_features = 576
        # print(self.in_features)
        # print([f"{b}\n\n\n New Layer" for b in self.backbone.blocks])
        self.head = nn.Sequential(
            nn.Conv1d(self.in_features, self.in_features//4, 3),
            nn.Linear(self.in_features, self.in_features//4),
            nn.Linear(self.in_features//4, self.in_features//16),

        )    
        self.out = nn.Linear(self.in_features//16, num_class)
        self.backbone.head = nn.Linear(self.backbone.head.in_features, num_class)
    def forward(self, x):
        # x1 = self.backbone.patch_embed(x)
        # x2 = self.backbone.pos_drop(x1)
        # x3 = self.blocks(x2)
        # x4 = self.backbone.norm(x3)
        # out = self.backbone.head(x4)
        # # out1 = out.view(out.size(0), -1)
        # print(out.size())
        # # out1 = self.head(out1)
        # print([i.size() for i in[x, x1, x2, x3, x4, out]])
        out = self.backbone(x)
        # print(out.size())
        return out



