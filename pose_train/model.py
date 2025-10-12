import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SpatialAttention(nn.Module):
    """Spatial attention module for focusing on important regions."""
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv2 = nn.Conv2d(in_channels // 8, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        attention = self.conv1(x)
        attention = F.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        return x * attention


class ChannelAttention(nn.Module):
    """Channel attention module for feature refinement."""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(in_channels)
        
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class PoseHead(nn.Module):
    """Specialized pose regression head with separate branches for translation and rotation."""
    def __init__(self, in_features, hidden_dim=256):
        super().__init__()
        self.translation_head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3)  # tx, ty, tz
        )
        
        self.rotation_head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 4)  # qw, qx, qy, qz
        )
        
    def forward(self, x):
        translation = self.translation_head(x)
        rotation = self.rotation_head(x)
        # Normalize quaternion
        rotation = F.normalize(rotation, p=2, dim=-1)
        return torch.cat([translation, rotation], dim=-1)


class AdvancedPoseNet(nn.Module):
    """Advanced CNN with attention mechanisms and specialized pose regression."""
    
    def __init__(self, in_channels=1, backbone='resnet18', pretrained=False, use_attention=True):
        super().__init__()
        self.backbone_name = backbone
        self.use_attention = use_attention
        
        if backbone == 'resnet18':
            res = models.resnet18(pretrained=pretrained)
            if in_channels != 3:
                if pretrained and hasattr(res, 'conv1'):
                    w = res.conv1.weight.data.clone()
                    w_mean = w.mean(dim=1, keepdim=True)
                    new_conv = nn.Conv2d(in_channels, w.shape[0], kernel_size=7, stride=2, padding=3, bias=False)
                    new_conv.weight.data = w_mean.repeat(1, in_channels, 1, 1)
                    res.conv1 = new_conv
                else:
                    res.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
            # Remove the final classification layer
            self.features = nn.Sequential(*list(res.children())[:-1])
            feat_dim = 512
            
            # Add attention modules
            if use_attention:
                self.attention = CBAM(512)
            
            # Use specialized pose head
            self.pose_head = PoseHead(feat_dim)
            
        elif backbone == 'resnet50':
            res = models.resnet50(pretrained=pretrained)
            if in_channels != 3:
                if pretrained and hasattr(res, 'conv1'):
                    w = res.conv1.weight.data.clone()
                    w_mean = w.mean(dim=1, keepdim=True)
                    new_conv = nn.Conv2d(in_channels, w.shape[0], kernel_size=7, stride=2, padding=3, bias=False)
                    new_conv.weight.data = w_mean.repeat(1, in_channels, 1, 1)
                    res.conv1 = new_conv
                else:
                    res.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
            self.features = nn.Sequential(*list(res.children())[:-1])
            feat_dim = 2048
            
            if use_attention:
                self.attention = CBAM(2048)
            
            self.pose_head = PoseHead(feat_dim)
            
        else:
            # Enhanced small CNN with attention
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 32, 7, stride=2, padding=3),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(32, 64, 5, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(128, 256, 3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            
            if use_attention:
                self.attention = CBAM(256)
            
            self.pose_head = PoseHead(256)

    def forward(self, x):
        f = self.features(x)
        
        if self.use_attention:
            f = self.attention(f)
        
        # Flatten for pose head
        f = F.adaptive_avg_pool2d(f, (1, 1))
        f = torch.flatten(f, 1)
        
        return self.pose_head(f)


class MultiAgentPoseNet(nn.Module):
    """Multi-agent pose estimation network for formation flying."""
    
    def __init__(self, in_channels=1, backbone='resnet18', pretrained=False, 
                 max_agents=5, use_attention=True):
        super().__init__()
        self.max_agents = max_agents
        
        # Shared feature extractor
        self.feature_extractor = AdvancedPoseNet(
            in_channels=in_channels, 
            backbone=backbone, 
            pretrained=pretrained,
            use_attention=use_attention
        )
        
        # Agent-specific pose heads
        self.agent_heads = nn.ModuleList([
            PoseHead(512 if backbone in ['resnet18'] else 2048 if backbone == 'resnet50' else 256)
            for _ in range(max_agents)
        ])
        
        # Agent detection head (binary classification for each agent)
        self.agent_detector = nn.Sequential(
            nn.Linear(512 if backbone in ['resnet18'] else 2048 if backbone == 'resnet50' else 256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, max_agents)
        )
        
    def forward(self, x):
        # Extract shared features
        features = self.feature_extractor.features(x)
        
        if self.feature_extractor.use_attention:
            features = self.feature_extractor.attention(features)
        
        # Global average pooling
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        
        # Detect which agents are present
        agent_presence = torch.sigmoid(self.agent_detector(features))
        
        # Predict poses for each agent
        agent_poses = []
        for i in range(self.max_agents):
            pose = self.agent_heads[i](features)
            # Mask poses based on agent presence
            pose = pose * agent_presence[:, i:i+1]
            agent_poses.append(pose)
        
        return torch.stack(agent_poses, dim=1), agent_presence


class SimplePoseNet(nn.Module):
    """A small CNN that regresses translation + quaternion from one or more grayscale images.

    Input shape: (B, C, H, W) where C is number of cameras (treated as channels).
    Output: 7-d vector [tx,ty,tz, qw, qx, qy, qz]
    """

    def __init__(self, in_channels=1, backbone='small', pretrained=False):
        super().__init__()
        self.backbone_name = backbone
        if backbone == 'resnet18':
            # create a ResNet backbone that accepts in_channels by modifying the first conv
            res = models.resnet18(pretrained=pretrained)
            # if using pretrained weights but input channels differ from 3, initialize conv1 by
            # averaging the pretrained weights across color channels to approximate grayscale.
            if in_channels != 3:
                if pretrained and hasattr(res, 'conv1'):
                    # get pretrained weights: shape (out, in, k, k)
                    w = res.conv1.weight.data.clone()
                    # average over the input channel dimension to produce single-channel weights
                    w_mean = w.mean(dim=1, keepdim=True)
                    # create new conv with desired in_channels and copy averaged weights
                    new_conv = nn.Conv2d(in_channels, w.shape[0], kernel_size=7, stride=2, padding=3, bias=False)
                    # repeat averaged weights to match in_channels
                    new_conv.weight.data = w_mean.repeat(1, in_channels, 1, 1)
                    res.conv1 = new_conv
                else:
                    # no pretrained weights or conv1 absent: replace conv1 with random init
                    res.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.features = nn.Sequential(*list(res.children())[:-1])
            feat_dim = 512
            self.regressor = nn.Sequential(
                nn.Flatten(),
                nn.Linear(feat_dim, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 7),
            )
        else:
            # small built-in CNN
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 32, 7, stride=2, padding=3),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 5, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.regressor = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 7),
            )

    def forward(self, x):
        f = self.features(x)
        out = self.regressor(f)
        return out
