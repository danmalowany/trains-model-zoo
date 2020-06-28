import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.resnet import resnet50, resnet101, resnet152
from torchvision.models.vgg import vgg16
from SSD.ssd_model import SSD
from SSD.multibox_loss import SSDLoss
from typing import List, Dict, Optional
from torch.tensor import Tensor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import RegionProposalNetwork
from torchvision.ops import boxes as box_ops
from trains import Task


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


def update_task_configuration(dict):
    configuration_data = Task.current_task().get_model_config_dict()
    for key, val in dict.items():
        configuration_data[key] = val
    Task.current_task().set_model_config(config_dict=configuration_data)


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


def get_model_instance_segmentation(num_classes, hidden_layer):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model


def freeze_layers(model, last_layer_to_freeze):
    for child in model.named_children():
        print('freezing layer {}'.format(child[0]))
        for param in child[1].parameters():
            param.requires_grad = False
        if child[0] == last_layer_to_freeze:
            break
    return model


def get_fasterrcnn(num_classes):
    class RegionProposalNetworkFixed():

        def assign_targets_to_anchors(self, anchors, targets):
            # type: (List[Tensor], List[Dict[str, Tensor]])
            labels = []
            matched_gt_boxes = []
            for anchors_per_image, targets_per_image in zip(anchors, targets):
                gt_boxes = targets_per_image["boxes"]

                if gt_boxes.numel() == 0:
                    # Background image (negative example)
                    device = anchors_per_image.device
                    matched_gt_boxes_per_image = torch.zeros_like(anchors_per_image, dtype=torch.float32, device=device)
                    labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
                else:
                    match_quality_matrix = box_ops.box_iou(gt_boxes, anchors_per_image)
                    matched_idxs = self.proposal_matcher(match_quality_matrix)
                    # get the targets corresponding GT for each proposal
                    # NB: need to clamp the indices because we can have a single
                    # GT in the image, and matched_idxs can be -2, which goes
                    # out of bounds
                    matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

                    labels_per_image = matched_idxs >= 0
                    labels_per_image = labels_per_image.to(dtype=torch.float32)

                    # Background (negative examples)
                    bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
                    labels_per_image[bg_indices] = torch.tensor(0.0)

                    # discard indices that are between thresholds
                    inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
                    labels_per_image[inds_to_discard] = torch.tensor(-1.0)

                labels.append(labels_per_image)
                matched_gt_boxes.append(matched_gt_boxes_per_image)
            return labels, matched_gt_boxes

    class RoIHeadsFixed():

        def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
            # type: (List[Tensor], List[Tensor], List[Tensor])
            matched_idxs = []
            labels = []
            for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):

                if gt_boxes_in_image.numel() == 0:
                    # Background image
                    device = gt_boxes_in_image.device
                    clamped_matched_idxs_in_image = torch.zeros(
                        (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                    )
                    labels_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
                else:
                    #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                    match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                    matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                    clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                    labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                    labels_in_image = labels_in_image.to(dtype=torch.int64)

                    # Label background (below the low threshold)
                    bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                    labels_in_image[bg_inds] = torch.tensor(0)

                    # Label ignore proposals (between low and high thresholds)
                    ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                    labels_in_image[ignore_inds] = torch.tensor(-1)  # -1 is ignored by sampler

                matched_idxs.append(clamped_matched_idxs_in_image)
                labels.append(labels_in_image)
            return matched_idxs, labels

        def select_training_samples(self, proposals, targets):
            # type: (List[Tensor], Optional[List[Dict[str, Tensor]]])
            self.check_targets(targets)
            assert targets is not None
            dtype = proposals[0].dtype

            gt_boxes = [t["boxes"].to(dtype) for t in targets]
            gt_labels = [t["labels"] for t in targets]

            # append ground-truth bboxes to propos
            proposals = self.add_gt_proposals(proposals, gt_boxes)

            # get matching gt indices for each proposal
            matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
            # sample a fixed proportion of positive-negative proposals
            sampled_inds = self.subsample(labels)
            matched_gt_boxes = []
            num_images = len(proposals)
            for img_id in range(num_images):
                img_sampled_inds = sampled_inds[img_id]
                proposals[img_id] = proposals[img_id][img_sampled_inds]
                labels[img_id] = labels[img_id][img_sampled_inds]
                matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

                gt_boxes_in_image = gt_boxes[img_id]
                if gt_boxes_in_image.numel() == 0:
                    device = gt_boxes_in_image.device
                    gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
                matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

            regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
            return proposals, matched_idxs, labels, regression_targets

    RegionProposalNetwork.assign_targets_to_anchors = RegionProposalNetworkFixed.assign_targets_to_anchors
    RoIHeads.assign_targets_to_proposals = RoIHeadsFixed.assign_targets_to_proposals
    RoIHeads.select_training_samples = RoIHeadsFixed.select_training_samples
    model = fasterrcnn_resnet50_fpn(pretrained_backbone=True, num_classes=num_classes)
    return model


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
