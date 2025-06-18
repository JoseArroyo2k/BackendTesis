import torch  # Añade esto en la primera línea del archivo
import torch.nn as nn
import timm

# 🔹 Bloque SE (Squeeze-and-Excitation)
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# 🔹 CNN Local con SEBlocks
class CNNLocalExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            SEBlock(32),
            nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            SEBlock(64),
            nn.MaxPool2d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SEBlock(128),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x.view(x.size(0), -1)

# 🔹 Atención multiescala
class MultiScaleAttention(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.Linear(in_features, in_features // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // 4, in_features),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Linear(in_features, in_features // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // 4, in_features),
            nn.Tanh()
        )

    def forward(self, x):
        ca = self.channel_attention(x)
        sa = self.spatial_attention(x)
        return x * ca + x * sa

# 🔹 Vision Transformer
class EnhancedVisionTransformer(nn.Module):
    def __init__(self, model_name="vit_small_patch16_224"):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=True)
        self.vit.head = nn.Identity()

        # Congelar embedding layer
        ct = 0
        for child in self.vit.children():
            ct += 1
            if ct < 2:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, x):
        return self.vit(x)

# 🔹 Modelo Final
class EnhancedTransMixNetDR(nn.Module):
    def __init__(self, num_classes=5, use_clinical_features=False, dropout=0.3, model_name="vit_small_patch16_224"):
        super().__init__()
        self.local_cnn = CNNLocalExtractor()
        self.global_vit = EnhancedVisionTransformer(model_name=model_name)

        vit_output_dim = 384 if "small" in model_name else 192
        self.local_encoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(vit_output_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.use_clinical_features = use_clinical_features
        if use_clinical_features:
            self.clinical_encoder = ClinicalFeaturesEncoder()
            self.fusion_dim = 256 + 256 + 64
        else:
            self.fusion_dim = 256 + 256

        self.fusion_attention = MultiScaleAttention(self.fusion_dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, clinical_features=None):
        x_local = self.local_cnn(x)
        x_local = self.local_encoder(x_local)

        x_global = self.global_vit(x)
        x_global = self.global_encoder(x_global)

        if self.use_clinical_features and clinical_features is not None:
            x_clinical = self.clinical_encoder(clinical_features)
            x_combined = torch.cat((x_local, x_global, x_clinical), dim=1)
        else:
            x_combined = torch.cat((x_local, x_global), dim=1)

        x_attended = self.fusion_attention(x_combined)
        return self.classifier(x_attended)

# 🔹 Codificador clínico (si se usa)
class ClinicalFeaturesEncoder(nn.Module):
    def __init__(self, num_features=5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64)
        )

    def forward(self, x):
        return self.encoder(x)
