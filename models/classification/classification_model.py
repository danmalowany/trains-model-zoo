from torchvision.models import vgg16, vgg19, resnet50, resnet101, resnet152, mobilenet_v2


def get_classifier(classifier_name):
    if classifier_name == 'vgg16':
        return vgg16(pretrained=True)
    elif classifier_name == 'vgg19':
        return vgg19(pretrained=True)
    elif classifier_name == 'resnet50':
        return resnet50(pretrained=True)
    elif classifier_name == 'resnet101':
        return resnet101(pretrained=True)
    elif classifier_name == 'resnet152':
        return resnet152(pretrained=True)
    elif classifier_name == 'mobilenet':
        return mobilenet_v2(pretrained=True)
    else:
        raise ValueError('Only "mobilenet", "vgg16", "vgg19", "resnet50", "resnet101" and "resnet152" are supported')
