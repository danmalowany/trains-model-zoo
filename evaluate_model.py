import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from dataclasses import dataclass, field

import torch
from ignite.engine import Events
from pathlib2 import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from trains import Task, InputModel

from datasets import CocoMask
from engines import create_evaluator
from events import evaluation_started, eval_iteration_completed, evaluation_completed
from models.utilities import get_model, get_iou_types
from torchvision_references.coco_eval import CocoEvaluator
from torchvision_references.coco_utils import convert_to_coco_api
from transforms import get_augmentations
from utilities import safe_collate


@dataclass
class EvalModel:
    """Class for keeping model and related information."""
    model: torch.nn.Module
    labels_enumeration: dict = field(default_factory=dict)
    model_configuration: dict = field(default_factory=dict)
    reports_prefix: str = ''


def run(task_args, external_model: EvalModel = None):
    if external_model is None:
        task = Task.init(project_name='Trains Model Zoo', task_name='Evaluate with PyTorch ecosystem')

        # Load a pretrained model and reset final fully connected layer.
        input_model = InputModel.import_model(weights_url='file:///tmp/checkpoints/model_epoch_20.pth')
        input_model.connect(task)
        input_checkpoint_path = input_model.get_weights()
        print('Loading model...')
        input_checkpoint = torch.load(input_checkpoint_path)

        model_configuration_data = input_checkpoint['configuration']
        labels_enumeration = input_checkpoint['labels_enumeration']

        label_enum = {val['name']: key for key, val in labels_enumeration.items()}
        num_classes = max([val for key, val in label_enum.items()]) + 1

        # Load a pretrained model and reset final fully connected layer.
        model = get_model(model_type=model_configuration_data.get('model_type'),
                          backbone_type=model_configuration_data.get('backbone'),
                          num_classes=num_classes,
                          transfer_learning=False,
                          configuration_data=model_configuration_data)
        model.load_state_dict(input_checkpoint['model'])

        reports_prefix = None
    else:
        model = external_model.model
        labels_enumeration = external_model.labels_enumeration
        model_configuration_data = external_model.model_configuration
        reports_prefix = external_model.reports_prefix

    # Set the training device to GPU if available - if not set it to CPU
    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.benchmark = True if torch.cuda.is_available() else False  # optimization for fixed input size

    # if there is more than one GPU, parallelize the model
    if torch.cuda.device_count() > 1:
        print("{} GPUs were detected - we will use all of them".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    model.to(device)

    # Define test dataset
    iou_types = get_iou_types(model)
    use_mask = True if "segm" in iou_types else False
    dataset = CocoMask(
        root=Path.joinpath(Path(task_args.val_dataset_ann_file).parent.parent,
                           task_args.val_dataset_ann_file.split('_')[1].split('.')[0]),
        annFile=task_args.val_dataset_ann_file,
        transforms=get_augmentations(train=False, image_size=model_configuration_data.get('image_size')),
        use_mask=use_mask)
    val_loader = DataLoader(dataset, batch_size=task_args.batch_size, shuffle=False, num_workers=4,
                            collate_fn=safe_collate, pin_memory=True)

    coco_api_val_dataset = convert_to_coco_api(dataset)

    writer = SummaryWriter(log_dir=task_args.log_dir)

    # define Ignite's evaluation engine
    evaluator = create_evaluator(model, device)

    @evaluator.on(Events.STARTED)
    def on_evaluation_started(engine):
        model.eval()
        engine.state.test_score_thr = task_args.test_score_thr
        engine.state.coco_evaluator = CocoEvaluator(coco_api_val_dataset, iou_types)
        engine.state.label_enum = {key: val['name'] for key, val in labels_enumeration.items()}
        evaluation_started(engine)

    @evaluator.on(Events.ITERATION_COMPLETED)
    def on_eval_iteration_completed(engine):
        eval_iteration_completed(engine, writer, task_args, prefix=reports_prefix)

    @evaluator.on(Events.COMPLETED)
    def on_evaluation_completed(engine):
        evaluation_completed(engine, writer, task_args, prefix=reports_prefix)

    evaluator.run(val_loader)
    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', type=int, default=6,
                        help='input batch size for training and validation')
    parser.add_argument('--log_interval', type=int, default=200,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--debug_images_interval', type=int, default=500,
                        help='how many batches to wait before logging debug images')
    parser.add_argument("--log_dir", type=str, default="/tmp/tensorboard_logs",
                        help="log directory for Tensorboard log output")
    parser.add_argument("--test_score_thr", nargs='*', type=float, default=[0.3, 0.5, 0.7],
                        help="Score threshold for evaluation")
    parser.add_argument('--val_dataset_ann_file', type=str,
                        default='/home/sam/Datasets/COCO2017/annotations/instances_val2017.json',
                        help='annotation file of test dataset')
    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    run(args)
