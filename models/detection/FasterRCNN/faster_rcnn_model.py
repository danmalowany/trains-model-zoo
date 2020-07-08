from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def get_fasterrcnn(num_classes, backbone_name='resnet50'):
    if backbone_name not in ['resnet50', 'resnet101', 'resnet152']:
        raise ValueError('Only "resnet50", "resnet101" and "resnet152" are supported backbone names')

    model = fasterrcnn_fpn(backbone=backbone_name, pretrained_backbone=True, num_classes=num_classes)
    return model


def fasterrcnn_fpn(backbone='resnet50', num_classes=91, pretrained_backbone=True, **kwargs):
    backbone = resnet_fpn_backbone(backbone, pretrained_backbone)
    model = FasterRCNN(backbone, num_classes, **kwargs)
    return model
