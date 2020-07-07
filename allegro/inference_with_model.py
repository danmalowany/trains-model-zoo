import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from functools import partial

import torch
from ignite.engine import Events
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from engines import create_annotator
from allegro.datasets import AllegroDataset
from transforms import get_augmentations
from models import get_model
from utilities import safe_collate

from allegroai import DataView, Task, InputModel
task = Task.init(project_name='Gun Detection', task_name='Inference with model',
                 output_uri='s3://allegro-private-datasets-restricted/YahooJP')

configuration_data = {'negative_frame_label': 'modelgun_nogood', 'positive_frame_label': 'modelgun_good'}
configuration_data = task.connect_configuration(configuration_data)


def run(task_args):
    # Set the training device to GPU if available - if not set it to CPU
    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.benchmark = True if torch.cuda.is_available() else False  # optimization for fixed input size

    input_model = InputModel.import_model(weights_url='s3://allegro-tutorials/Gun Detection/Train fasterrcnn with torchvision - SM - ResNet50 - layer2 - mix.01cb3cb2853b4923a5d9fa16f9032823/models/model_epoch_48.pth')
    input_checkpoint_path = input_model.get_weights()
    print('Loading model...')
    input_checkpoint = torch.load(input_checkpoint_path, map_location=torch.device(device))
    model_configuration_data = input_checkpoint['configuration']
    labels_enumeration = input_checkpoint['labels_enumeration']
    num_classes = len([key for key, val in labels_enumeration.items() if val >= 0])

    val_dv = DataView()
    val_dv.add_query(dataset_name='Data registration example', version_id='33b5bcab4a2545c68dd94311231dc073')
    target_version = val_dv.get_versions()
    if len(target_version) > 1:
        raise ValueError('Dataview should include only one dataset version')

    dataset = AllegroDataset(dataview=val_dv,
                             transforms_func=partial(get_augmentations,
                                                     image_size=model_configuration_data.get('image_size')),
                             train=False, annotate=True)
    val_loader = DataLoader(dataset, batch_size=task_args.batch_size, shuffle=False, num_workers=6,
                            collate_fn=safe_collate, pin_memory=True)

    # Load a pretrained model and reset final fully connected layer.
    model = get_model(model_type=model_configuration_data.get('model_type'),
                      backbone_type=model_configuration_data.get('backbone'),
                      num_classes=num_classes,
                      transfer_learning=False,
                      configuration_data=model_configuration_data)
    model.load_state_dict(input_checkpoint['model'])

    # if there is more than one GPU, parallelize the model
    if torch.cuda.device_count() > 1:
        print("{} GPUs were detected - we will use all of them".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    # copy the model to each device
    model.to(device)

    writer = SummaryWriter(log_dir=task_args.log_dir)

    # define Ignite's evaluation engine
    annotator = create_annotator(model, device)

    @annotator.on(Events.STARTED)
    def on_annotation_started(engine):
        engine.state.label_enum = {val: key for key, val in labels_enumeration.items()}
        engine.state.frame_labels = {'negative': configuration_data.get('negative_frame_label', 'negative'),
                                     'positive': configuration_data.get('positive_frame_label', 'positive')}
        engine.state.version = target_version[0]
        engine.state.frames = []
        model.eval()

    @annotator.on(Events.ITERATION_COMPLETED)
    def on_annotation_iteration_completed(engine):
        inputs, preds, frames = engine.state.output
        engine.state.frames.extend(frames)

        if engine.state.iteration % task_args.log_interval == 0:
            print("Evaluation: Iteration: {}".format(engine.state.iteration))

        if engine.state.iteration % task_args.debug_images_interval == 0:
            for n, data_tuple in enumerate(zip(inputs, preds)):
                print('need to implement debug image')
                # debug_image = draw_debug_images(data_tuple[0].cpu())
                # cls = engine.state.label_enum[int(data_tuple[1])]
                # writer.add_image("annotation/{}_{} pred_{}".format(engine.state.iteration, n, cls),
                #                  debug_image, engine.state.iteration, dataformats='HWC')

    @annotator.on(Events.COMPLETED)
    def on_annotation_completed(engine):
        engine.state.version.update_frames(engine.state.frames)

    annotator.run(val_loader)
    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', type=int, default=6,
                        help='input batch size for training and validation')
    parser.add_argument('--log_interval', type=int, default=200,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--debug_images_interval', type=int, default=1000,
                        help='how many batches to wait before logging debug images')
    parser.add_argument("--log_dir", type=str, default="/tmp/tensorboard_logs",
                        help="log directory for Tensorboard log output")
    parser.add_argument("--test_score_thr", type=float, default=0.8,
                        help="Score threshold for debug images and frame level accuracy")
    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    run(args)
