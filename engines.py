import copy
import math

import torch
from trains import Task
from ignite.engine import Engine
from torchvision.transforms import functional as F

from torchvision_references import utils


def create_trainer(model, device):
    def update_model(engine, batch):
        # images, targets = copy.deepcopy(batch)
        images, targets = prepare_batch(copy.deepcopy(batch), device=torch.device('cpu'))
        images_model, targets_model = prepare_batch(batch, device=device)

        loss_dict = model(images_model, targets_model)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        engine.state.optimizer.zero_grad()
        if not math.isfinite(loss_value):
            print("Loss is {}, resetting loss and skipping training iteration".format(loss_value))
            print('Loss values were: ', loss_dict_reduced)
            print('Input labels were: ', [target['labels'] for target in targets])
            print('Input boxes were: ', [target['boxes'] for target in targets])
            loss_dict_reduced = {k: torch.tensor(0) for k, v in loss_dict_reduced.items()}
        else:
            losses.backward()
            engine.state.optimizer.step()

        if engine.state.warmup_scheduler is not None:
            engine.state.warmup_scheduler.step()

        if type(engine.state.scheduler) == torch.optim.lr_scheduler.CyclicLR:
            engine.state.scheduler.step()

        images_model = targets_model = None
        loss_dict_reduced_detach = {key: val.cpu() for key, val in loss_dict_reduced.items()}

        return images, targets, loss_dict_reduced_detach
    return Engine(update_model)


def create_evaluator(model, device):
    def update_model(engine, batch):
        images, targets = prepare_batch(batch, device=device)
        images_model = copy.deepcopy(images)

        torch.cuda.synchronize()
        with torch.no_grad():
            outputs = model(images_model)

        outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]

        res = {target["image_id"].item(): output if len(output['boxes']) > 0 else {}
               for target, output in zip(targets, outputs)}
        engine.state.coco_evaluator.update(res)

        # Frame level statistics
        task_params = Task.current_task().data.execution.parameters
        target_labels = [target['labels'].cpu() for target in targets]
        pred_labels = [val['labels'][val['scores'] > float(task_params['test_score_thr'])].cpu()
                       if len(val)>0 else torch.tensor([], dtype=torch.int64) for key, val in res.items()]
        iter_corrects = [torch.all(torch.eq(target.unique(), pred.unique())).item()
                              if target.unique().size() == pred.unique().size() else False
                              for target, pred in zip(target_labels, pred_labels)]

        images_model = outputs = None

        return images, targets, res, pred_labels, iter_corrects
    return Engine(update_model)


def create_annotator(model, device):
    def update_model(engine, batch):
        images, frames = batch
        images = list(F.to_tensor(F.to_pil_image(image)).to(device, non_blocking=True) for image in images)
        images_model = copy.deepcopy(images)

        torch.cuda.synchronize()
        with torch.no_grad():
            outputs = model(images_model)

        outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]

        res = {frame.id: output if len(output['boxes']) > 0 else {}
               for frame, output in zip(frames, outputs)}

        for image, frame, result in zip(images, frames, res.items()):
            if len(result[1]) == 0:
                frame.add_annotation(frame_class=[engine.state.frame_labels['negative']], confidence=1.0)
            else:
                confidence_list = []
                # Apply score threshold
                task_params = Task.current_task().data.execution.parameters
                for score, label_id, box in zip(result[1]['scores'], result[1]['labels'], result[1]['boxes']):
                    if float(score.cpu()) > float(task_params['test_score_thr']):
                        label_id = int(label_id.cpu())
                        label = engine.state.label_enum[label_id]
                        d1, d2, d3 = image.shape
                        resize_box_factors = [frame.width / d3, frame.height / d2] * 2
                        resized_box = box.cpu() * torch.tensor(resize_box_factors)
                        box_xywh = resized_box[:2].tolist() + (resized_box[2:] - resized_box[:2]).tolist()
                        confidence = float(score.cpu())
                        confidence_list.append(confidence)
                        frame.add_annotation(labels=[label], box2d_xywh=box_xywh, confidence=confidence)

                if len(confidence_list) == 0:
                    confidence_list = [score for score in result[1]['scores']]
                    frame.add_annotation(frame_class=[engine.state.frame_labels['negative']],
                                         confidence=1 - max(confidence_list))
                else:
                    frame.add_annotation(frame_class=[engine.state.frame_labels['positive']],
                                         confidence=max(confidence_list))

        return images, res, frames
    return Engine(update_model)


def prepare_batch(batch, device=None):
    images, targets = batch
    images = list(F.to_tensor(F.to_pil_image(image)).to(device, non_blocking=True) for image in images)
    targets = [{k: torch.as_tensor(v).to(device, non_blocking=True) for k, v in t.items()} for t in targets]
    # images = list(image.to(device, non_blocking=True) for image in images)
    # targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
    return images, targets
