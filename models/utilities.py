import torch
import torchvision

from utilities import update_task_configuration
from models.detection.FasterRCNN.faster_rcnn_model import get_fasterrcnn
from models.detection.SSD.backbones import get_backbone
from models.detection.SSD.multibox_loss import SSDLoss
from models.detection.SSD.ssd_model import SSD
from models.segmentation.MaskRCNN.mask_rcnn_model import get_model_instance_segmentation


def get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


def get_model(model_type, backbone_type, num_classes, dropout=0, transfer_learning=True, configuration_data=None):
    if model_type == 'maskrcnn':
        mask_predictor_hidden_layer = configuration_data.get('mask_predictor_hidden_layer', 256)
        model = get_model_instance_segmentation(num_classes, mask_predictor_hidden_layer)
        last_layer_to_freeze = configuration_data.get('last_layer_to_freeze', 'layer3')
        update_task_configuration({'mask_predictor_hidden_layer': mask_predictor_hidden_layer,
                                   'last_layer_to_freeze': last_layer_to_freeze})
        if transfer_learning:
            model.backbone.body = freeze_layers(model=model.backbone.body, last_layer_to_freeze=last_layer_to_freeze)
    elif model_type == 'fasterrcnn':
        model = get_fasterrcnn(num_classes)
        last_layer_to_freeze = configuration_data.get('last_layer_to_freeze', 'layer3')
        update_task_configuration({'last_layer_to_freeze': last_layer_to_freeze})
        if transfer_learning:
            model.backbone.body = freeze_layers(model=model.backbone.body, last_layer_to_freeze=last_layer_to_freeze)
    elif model_type == 'ssd':
        backbone = get_backbone(backbone_type)
        model = SSD(backbone=backbone, num_classes=num_classes, dropout=dropout,
                    loss_function=SSDLoss(num_classes))
        model.change_input_size(
            torch.rand(size=(1, 3, configuration_data.get('image_size'), configuration_data.get('image_size'))) * 255)
        model.box_coder.enforce_matching = configuration_data.get('enforce_matching', True)
        last_layer_to_freeze = configuration_data.get('last_layer_to_freeze', 'layer2')
        # last_layer_to_freeze = configuration_data.get('last_layer_to_freeze', 'till_conv4_3')
        update_task_configuration({'enforce_matching': model.box_coder.enforce_matching,
                                   'last_layer_to_freeze': last_layer_to_freeze})
        if transfer_learning:
            if last_layer_to_freeze in [child[0] for child in model.extractor.till_conv4_3.named_children()]:
                model.extractor.till_conv4_3 = freeze_layers(model=model.extractor.till_conv4_3,
                                                             last_layer_to_freeze=last_layer_to_freeze)
            elif last_layer_to_freeze in [child[0] for child in model.extractor.till_conv5_3.named_children()]:
                model.extractor.till_conv5_3 = freeze_layers(model=model.extractor.till_conv5_3,
                                                             last_layer_to_freeze=last_layer_to_freeze)
            else:
                raise ValueError('layer "{}" was not found'.format(last_layer_to_freeze))
            # model.extractor = freeze_layers(model=model.extractor, last_layer_to_freeze=last_layer_to_freeze)
    else:
        raise ValueError('Only "maskrcnn" and "ssd" are supported as model type')
    return model


def freeze_layers(model, last_layer_to_freeze):
    for child in model.named_children():
        print('freezing layer {}'.format(child[0]))
        for param in child[1].parameters():
            param.requires_grad = False
        if child[0] == last_layer_to_freeze:
            break
    return model
