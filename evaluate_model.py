import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
import torch.nn as nn
from ignite.engine import Events
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

from engines import create_evaluator
from events import evaluation_started, eval_iteration_completed, evaluation_completed
from utils import AllegroDataset, safe_collate, get_transforms

from allegroai import DataView, Task, InputModel
task = Task.init(project_name='NSFW Image Classification', task_name='Evaluate with torchvision',
                 output_uri='s3://allegro-private-datasets-restricted/YahooJP')


def run(task_args):
    # Set the training device to GPU if available - if not set it to CPU
    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.benchmark = True if torch.cuda.is_available() else False  # optimization for fixed input size

    # Load a pretrained model and reset final fully connected layer.
    input_model = InputModel.import_model(weights_url='s3://allegro-private-datasets-restricted/YahooJP/NSFW Image Classification/Final train ResNet50 - CyclilcLR - DO 0.45 - layer 2 - mix - more aug - 0.01 - more hard.49700fb0f943498587b2ded3b7db5663/models/model_epoch_110.pth')
    input_model.connect(task)
    input_checkpoint_path = input_model.get_weights()
    print('Loading model...')
    input_checkpoint = torch.load(input_checkpoint_path, map_location=torch.device(device))
    configuration_data = input_checkpoint['configuration']
    labels_enumeration = input_checkpoint['labels_enumeration']

    # Define train and test datasets
    val_dv = DataView(name='val')
    val_dv.add_query(dataset_name='NSFW Test w/o specific labels', version_id='69c890da5bfd45d08b8018d463a3e54b')
    val_dv.set_labels(labels_enumeration)

    dataset = AllegroDataset(dataview=val_dv, transforms=get_transforms(False, configuration_data.get('image_size')),
                             train=False)
    val_loader = DataLoader(dataset, batch_size=task_args.batch_size, shuffle=False, num_workers=4,
                            collate_fn=safe_collate, pin_memory=True)

    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    num_classes = len(labels_enumeration)
    model.fc = nn.Sequential(*[nn.Dropout(p=configuration_data.get('dropout')), nn.Linear(num_ftrs, num_classes)])
    model.load_state_dict(input_checkpoint['model'])

    # if there is more than one GPU, parallelize the model
    if torch.cuda.device_count() > 1:
        print("{} GPUs were detected - we will use all of them".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    # copy the model to each device
    model.to(device)

    writer = SummaryWriter(log_dir=task_args.log_dir)

    # define Ignite's evaluation engine
    evaluator = create_evaluator(model, device)

    @evaluator.on(Events.STARTED)
    def on_evaluation_started(engine):
        engine.state.test_score_thr = task_args.test_score_thr
        model.eval()
        evaluation_started(engine)

    @evaluator.on(Events.ITERATION_COMPLETED)
    def on_eval_iteration_completed(engine):
        eval_iteration_completed(engine, writer, task_args)

    @evaluator.on(Events.COMPLETED)
    def on_evaluation_completed(engine):
        evaluation_completed(engine, writer, task_args)

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
    parser.add_argument("--test_score_thr", nargs='*', type=float, default=[0.4, 0.5, 0.6],
                        help="Score threshold for debug images and frame level accuracy")
    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    run(args)
