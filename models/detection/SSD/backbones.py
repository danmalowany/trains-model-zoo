from torchvision.models import vgg16, resnet50, resnet101, resnet152


def get_backbone(backbone_name):
    if backbone_name == 'vgg16':
        return vgg16(pretrained=True)
    elif backbone_name == 'resnet50':
        return resnet50(pretrained=True)
    elif backbone_name == 'resnet101':
        return resnet101(pretrained=True)
    elif backbone_name == 'resnet152':
        return resnet152(pretrained=True)
    else:
        raise ValueError('Only "vgg16", "resnet50", "resnet101" and "resnet152" are supported backbone names')
