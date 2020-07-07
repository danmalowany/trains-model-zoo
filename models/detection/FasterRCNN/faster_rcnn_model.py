from torchvision.models.detection import fasterrcnn_resnet50_fpn


def get_fasterrcnn(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained_backbone=True, num_classes=num_classes)
    return model
