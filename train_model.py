import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from functools import partial
from itertools import chain

import numpy as np
import torch
from ignite.engine import Events
from pathlib2 import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from trains import Task
from trains.storage.helper import StorageHelper

from datasets import CocoMask
from engines import create_trainer, create_evaluator
from events import evaluation_completed, eval_iteration_completed, evaluation_started
from models.detection.SSD.priorbox_optimization import PriorOptimizationOptions
from models.detection.SSD.priorbox_optimization.optimize_priors import optimize_priors
from models.detection.SSD.priorbox_optimization.priors_optimization_utils import collect_ground_truth_stats, \
    get_optimization_input, convert_optimization_result_to_priors
from models.utilities import get_model, get_iou_types
from torchvision_references import utils
from torchvision_references.coco_eval import CocoEvaluator
from torchvision_references.coco_utils import convert_to_coco_api
from transforms import get_augmentations
from utilities import draw_debug_images, draw_mask, safe_collate

task = Task.init(project_name='Trains Model Zoo',
                 task_name='Train with PyTorch ecosystem')

configuration_data = {'image_size': 512, 'model_type': 'ssd', 'backbone': 'resnet50'}
configuration_data = task.connect_configuration(configuration_data)


def get_data_loaders(train_ann_file, test_ann_file, batch_size, test_size, opt_size,
                     image_size, use_mask, workers, pin_memory):
    # first, crate PyTorch dataset objects, for the train and validation data.
    dataset = CocoMask(
        root=Path.joinpath(Path(train_ann_file).parent.parent, train_ann_file.split('_')[1].split('.')[0]),
        annFile=train_ann_file,
        transforms=get_augmentations(train=True, image_size=image_size),
        use_mask=use_mask)
    dataset_test = CocoMask(
        root=Path.joinpath(Path(test_ann_file).parent.parent, test_ann_file.split('_')[1].split('.')[0]),
        annFile=test_ann_file,
        transforms=get_augmentations(train=False, image_size=image_size),
        use_mask=use_mask)

    labels_enumeration = dataset.coco.cats

    indices_val = torch.randperm(len(dataset_test)).tolist()
    dataset_val = torch.utils.data.Subset(dataset_test, indices_val[:test_size])

    indices_opt = torch.randperm(len(dataset)).tolist()
    dataset_opt = torch.utils.data.Subset(dataset, indices_opt[:opt_size])

    # set train and validation data-loaders
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                              collate_fn=safe_collate, pin_memory=pin_memory)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=workers,
                            collate_fn=safe_collate, pin_memory=pin_memory)
    opt_loader = DataLoader(dataset_opt, batch_size=batch_size, shuffle=False, num_workers=workers,
                            collate_fn=safe_collate, pin_memory=False)

    return train_loader, val_loader, opt_loader, labels_enumeration


def run(task_args):
    num_classes = 91
    
    # Set the training device to GPU if available - if not set it to CPU
    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.benchmark = True if torch.cuda.is_available() else False  # optimization for fixed input size

    # Get the relevant model based in task arguments
    model = get_model(model_type=configuration_data.get('model_type'),
                      backbone_type=configuration_data.get('backbone'),
                      num_classes=num_classes,
                      dropout=task_args.dropout,
                      transfer_learning=task_args.transfer_learning,
                      configuration_data=configuration_data)

    # if there is more than one GPU, parallelize the model
    if torch.cuda.device_count() > 1:
        print("{} GPUs were detected - we will use all of them".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    # copy the model to each device
    model.to(device)
    
    # Define train and test datasets
    iou_types = get_iou_types(model)
    use_mask = True if "segm" in iou_types else False
    train_loader, val_loader, opt_loader, labels_enum = get_data_loaders(task_args.train_dataset_ann_file,
                                                             task_args.val_dataset_ann_file,
                                                             task_args.batch_size,
                                                             task_args.test_size,
                                                             task_args.optimize_priors_sample_size,
                                                             configuration_data.get('image_size'),
                                                             use_mask,
                                                             task_args.num_workers,
                                                             task_args.pin_memory)

    val_dataset = list(chain.from_iterable(zip(*batch) for batch in iter(val_loader)))
    coco_api_val_dataset = convert_to_coco_api(val_dataset)

    if task_args.input_checkpoint:
        storage_helper = StorageHelper.get(task.output_uri)
        input_checkpoint_path = storage_helper.get_local_copy(task_args.input_checkpoint)
        print('Loading model checkpoint from {}'.format(task_args.input_checkpoint))
        input_checkpoint = torch.load(input_checkpoint_path, map_location=torch.device(device))
        model.load_state_dict(input_checkpoint['model'])

    opt_result = None
    if (task_args.optimize_priors_sample_size > 0) and (configuration_data.get('model_type') == 'ssd'):
        print('Generating optimized priors based on {} data samples, using {} method'.format(
            task_args.optimize_priors_sample_size, task_args.optimization_method))
        # collect statistics
        ground_truth_statistics = collect_ground_truth_stats(opt_loader)

        if ground_truth_statistics is not None:
            # prior optimization options
            opt_options = PriorOptimizationOptions(
                target_size_w=configuration_data.get('image_size'),
                target_size_h=configuration_data.get('image_size'),
                cluster_threshold=0.9,
                optimization_method=args.optimization_method,
                plot_results=True,
                gen_match_report=False,
            )
            opt_input = get_optimization_input(ground_truth_statistics, model.fm_sizes,
                                               model.box_coder.priors, configuration_data.get('image_size'))
            # prior optimization and match report
            opt_result = optimize_priors(opt_input, opt_options)

    if 'priors_output' in configuration_data:
        if opt_result is None:
            from models.detection.SSD import PriorOptimizationOutput, ImageSizeTuple
            import pandas as pd
            print('Loading priors from {}'.format(configuration_data['priors_output']))
            storage_helper = StorageHelper.get(task.output_uri)
            priors_output_path = storage_helper.get_local_copy(configuration_data['priors_output'])
            image_size = configuration_data.get('image_size')
            target_image_size = ImageSizeTuple(w=image_size, h=image_size)
            priors_df = pd.read_csv(priors_output_path)
            opt_result = PriorOptimizationOutput(target_image_size, priors_df)

    if opt_result is not None:
        priors_generator = partial(convert_optimization_result_to_priors, opt_result=opt_result)
        model.update_priors(priors_generator=priors_generator)
        model.to(device)

    writer = SummaryWriter(log_dir=task_args.log_dir)
    
    # define Ignite's train and evaluation engine
    trainer = create_trainer(model, device)
    evaluator = create_evaluator(model, device)
    
    @trainer.on(Events.STARTED)
    def on_training_started(engine):
        engine.state.label_enum = {key: val['name'] for key, val in labels_enum.items()}
        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        engine.state.optimizer = torch.optim.SGD(params,
                                                 lr=task_args.lr,
                                                 momentum=task_args.momentum,
                                                 weight_decay=task_args.weight_decay)

        if task_args.scheduler_type == 'StepLR':
            cycles_number = np.log(0.001) / np.log(0.1)
            step_size = int((task_args.epochs / cycles_number))
            engine.state.scheduler = torch.optim.lr_scheduler.StepLR(engine.state.optimizer,
                                                                     step_size=step_size,
                                                                     gamma=0.1)
        elif task_args.scheduler_type == 'CyclicLR':
            cycles_number = np.log(0.001 * 8) / np.log(0.5)
            epoch_size = len(train_loader)
            step_size = int((task_args.epochs / cycles_number) * epoch_size * 0.5)
            engine.state.scheduler = torch.optim.lr_scheduler.CyclicLR(engine.state.optimizer,
                                                                       base_lr=task_args.lr*0.001,
                                                                       max_lr=task_args.lr,
                                                                       step_size_up=step_size,
                                                                       step_size_down=step_size,
                                                                       mode='triangular2')
        else:
            raise ValueError('Only "StepLR" and "CyclicLR" are supported as scheduler type')

        if type(engine.state.scheduler) == torch.optim.lr_scheduler.StepLR:
            engine.state.warmup_iters = max(task_args.warmup_iterations, len(train_loader) - 1)
            print('Warm up period was set to {} iterations'.format(engine.state.warmup_iters))
            warmup_factor = 1. / engine.state.warmup_iters
            engine.state.warmup_scheduler = utils.warmup_lr_scheduler(engine.state.optimizer,
                                                                      engine.state.warmup_iters, warmup_factor)

        if task_args.input_checkpoint and task_args.load_optimizer:
            engine.state.optimizer.load_state_dict(input_checkpoint['optimizer'])
            engine.state.scheduler.load_state_dict(input_checkpoint['lr_scheduler'])

        if task_args.min_checkpoint_iterations > 0:
            engine.state.last_epoch_iteration = 0

    @trainer.on(Events.EPOCH_STARTED)
    def on_epoch_started(engine):
        model.train()

    @trainer.on(Events.ITERATION_COMPLETED)
    def on_iteration_completed(engine):
        images, targets, loss_dict_reduced = engine.state.output
        if engine.state.iteration % task_args.log_interval == 0:
            loss = sum(loss for loss in loss_dict_reduced.values()).item()
            print("Epoch: {}, Iteration: {}, Loss: {}".format(engine.state.epoch, engine.state.iteration, loss))
            for k, v in loss_dict_reduced.items():
                writer.add_scalar("loss/{}".format(k), v.item(), engine.state.iteration)
            writer.add_scalar("loss/total_loss", sum(loss for loss in loss_dict_reduced.values()).item(), engine.state.iteration)
            writer.add_scalar("learning rate/lr", engine.state.optimizer.param_groups[0]['lr'], engine.state.iteration)

        if engine.state.iteration % task_args.debug_images_interval == 0:
            for n, debug_image in enumerate(draw_debug_images(images, targets, score_thr=task_args.test_score_thr,
                                                              labels_enum=engine.state.label_enum)):
                writer.add_image("training/image_{}".format(n), debug_image, engine.state.iteration, dataformats='HWC')
                if 'masks' in targets[n]:
                    writer.add_image("training/image_{}_mask".format(n),
                                     draw_mask(targets[n]), engine.state.iteration, dataformats='HW')
        images = targets = loss_dict_reduced = engine.state.output = None

    @trainer.on(Events.EPOCH_COMPLETED)
    def on_epoch_completed(engine):
        if engine.state.warmup_scheduler is not None and \
                engine.state.warmup_scheduler.last_epoch >= engine.state.warmup_iters:
            engine.state.warmup_scheduler = None

        if type(engine.state.scheduler) == torch.optim.lr_scheduler.StepLR and engine.state.warmup_scheduler is None:
            engine.state.scheduler.step()

        if (task_args.min_checkpoint_iterations == 0) or \
                (engine.state.iteration - engine.state.last_epoch_iteration) > task_args.min_checkpoint_iterations or \
                (engine.state.epoch == task_args.epochs):
            engine.state.last_epoch_iteration = engine.state.iteration

            checkpoint_path = os.path.join(task_args.output_dir, 'model_epoch_{}.pth'.format(engine.state.epoch))
            print('Saving checkpoint')
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': engine.state.optimizer.state_dict(),
                'lr_scheduler': engine.state.scheduler.state_dict(),
                'epoch': engine.state.epoch,
                'configuration': configuration_data,
                'labels_enumeration': labels_enum}
            utils.save_on_master(checkpoint, checkpoint_path)
            print('Checkpoint from epoch {} was saved at {}'.format(engine.state.epoch, checkpoint_path))

            print('Starting checkpoint evaluation')
            evaluator.run(val_loader)

            evaluator.state = checkpoint = None

    @evaluator.on(Events.STARTED)
    def on_evaluation_started(engine):
        model.eval()
        engine.state.test_score_thr = task_args.test_score_thr
        engine.state.coco_evaluator = CocoEvaluator(coco_api_val_dataset, iou_types)
        engine.state.label_enum = {key: val['name'] for key, val in labels_enum.items()}
        evaluation_started(engine)

    @evaluator.on(Events.ITERATION_COMPLETED)
    def on_eval_iteration_completed(engine):
        eval_iteration_completed(engine, writer, task_args, trainer.state.iteration)

    @evaluator.on(Events.COMPLETED)
    def on_evaluation_completed(engine):
        evaluation_completed(engine, writer, task_args, trainer.state.iteration)

    trainer.run(train_loader, max_epochs=task_args.epochs)
    writer.close()
    
    
if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--min_checkpoint_iterations', type=int, default=5000,
                        help='Skipping evaluation and checkpoint if epoch ends before "min_checkpoint_iterations"')
    parser.add_argument('--warmup_iterations', type=int, default=5000,
                        help='Number of iteration for warmup period (until reaching base learning rate)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='input batch size for training and validation')
    parser.add_argument('--test_size', type=int, default=2000,
                        help='number of frames from the test dataset to use for validation')
    parser.add_argument("--test_score_thr", nargs='*', type=float, default=[0.3, 0.5, 0.7],
                        help="Score threshold for evaluation")
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='number of sub-processes for DataLoader to use for data loading')
    parser.add_argument('--pin_memory', type=bool, default=True,
                        help='If True, the dataloader will copy Tensors into CUDA before returning them.')
    parser.add_argument('--log_interval', type=int, default=200,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--debug_images_interval', type=int, default=400,
                        help='how many batches to wait before logging debug images')
    parser.add_argument('--train_dataset_ann_file', type=str,
                        default='/home/sam/Datasets/COCO2017/annotations/instances_train2017.json',
                        help='annotation file of train dataset')
    parser.add_argument('--val_dataset_ann_file', type=str,
                        default='/home/sam/Datasets/COCO2017/annotations/instances_val2017.json',
                        help='annotation file of test dataset')
    parser.add_argument('--input_checkpoint', type=str, default='',
                        help='Loading model weights from this checkpoint.')
    parser.add_argument('--load_optimizer', default=False, type=bool,
                        help='Use optimizer and lr_scheduler saved in the input checkpoint to resume training')
    parser.add_argument('--scheduler_type', default='StepLR', type=str,
                        help='Sets the learning scheduler. we support "CyclicLR" or "StepLR" only')
    parser.add_argument('--transfer_learning', default=True, type=bool,
                        help='If True,freezing feature extractor layers (all layers beside the last fc layer)')
    parser.add_argument("--output_dir", type=str, default="/tmp/checkpoints",
                        help="output directory for saving models checkpoints")
    parser.add_argument("--log_dir", type=str, default="/tmp/tensorboard_logs",
                        help="log directory for Tensorboard log output")
    parser.add_argument('--optimization_method', type=str, default='Kmeans_per_feature_map',
                        help='Optimization method to use for prior optimization. We support: '
                             '"Kmeans_per_feature_map" and "Kmeans_global"')
    parser.add_argument('--optimize_priors_sample_size', type=int, default=0,
                        help='how many images should be used to optimize the bounding box prior sizes')
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate for optimizer")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="momentum for optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.0005,
                        help="weight decay for optimizer")
    parser.add_argument("--dropout", type=float, default=0.25,
                        help="dropout for model's Dropout layers")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        utils.mkdir(args.output_dir)
    if not os.path.exists(args.log_dir):
        utils.mkdir(args.log_dir)

    run(args)
