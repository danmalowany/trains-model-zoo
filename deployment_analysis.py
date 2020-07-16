from time import time

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision
import yaml
import os
import copy
from trains import Task, InputModel
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from models.utilities import get_model
from evaluate_model import run
from evaluate_model import EvalModel

task = Task.init(project_name='Trains Model Zoo', task_name='Deployment analysis with PyTorch ecosystem')


def evaluate_model(task_args):
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

    eval_model = EvalModel(model, labels_enumeration, model_configuration_data, 'Original model')
    run(task_args, eval_model)
    model.cpu()

    # # Dynamic model quantization
    quantized_model = copy.deepcopy(model)
    quantized_model = torch.quantization.quantize_dynamic(
        quantized_model, {nn.Conv2d, nn.Linear}, dtype=torch.qint8
    )

    input_checkpoint['model'] = quantized_model.state_dict()

    eval_model = EvalModel(quantized_model, labels_enumeration, model_configuration_data, 'Quantized model')
    run(task_args, eval_model)
    del quantized_model

    # Global pruning
    pruned_model = copy.deepcopy(model)
    parameters_to_prune = []
    for module in pruned_model.named_modules():
        if type(module[1]) in [nn.Conv2d, nn.Linear]:
            parameters_to_prune.append((module[0], module[1], 'weight'))

    prune.global_unstructured(
        [param[1:] for param in parameters_to_prune],
        pruning_method=prune.L1Unstructured,
        amount=task_args.pruning_amount,
    )

    for module in parameters_to_prune:
        print("Sparsity in {}: {:.2f}%".format(module[0],
                                               100. * float(torch.sum(module[1].weight == 0))
                                               / float(module[1].weight.nelement())
                                               )
              )

    print(
        "Global sparsity: {:.2f}%".format(
            100. * float(sum([torch.sum(module[1].weight == 0) for module in parameters_to_prune]))
            / float(sum([module[1].weight.nelement() for module in parameters_to_prune]))
        )
    )

    for module in parameters_to_prune:
        prune.remove(module[1], module[2])

    eval_model = EvalModel(pruned_model, labels_enumeration, model_configuration_data, 'Pruned model')
    run(task_args, eval_model)
    del pruned_model

    #
    # def model_evaluation(model_to_eval, test_data):
    #     def size_model_evaluation(eval_model):
    #         torch.save(eval_model.state_dict(), "temp.p")
    #         size = os.path.getsize("temp.p")/1e6
    #         os.remove('temp.p')
    #         return size
    #
    #     def time_model_evaluation(eval_model, testset):
    #         s = time()
    #         with torch.no_grad():
    #             eval_model(test_data)
    #         elapsed = time() - s
    #         return elapsed
    #
    #     model_size = size_model_evaluation(model_to_eval)
    #     inference_time = time_model_evaluation(model_to_eval, test_data)
    #
    #     return model_size, inference_time
    #
    # for model in [model, quantized_model, pruned_model]:
    #     model_size, inference_time = model_evaluation(model, example)
    #     print('Size (MB):', model_size)
    #     print('elapsed time (seconds): {:.1f}'.format(inference_time))
    #
    # # Use torch.jit.script to generate a torch.jit.ScriptModule via scripting.
    # scripted_model = torch.jit.script(model)
    #
    # path_to_c_example = '/tmp/deploy'
    # if not os.path.isdir(path_to_c_example):
    #     os.mkdir(path_to_c_example)
    # scripted_model.save(path_to_c_example+'/scripted_model.pt')
    # traced_ops = torch.jit.export_opnames(scripted_model)
    # with open(path_to_c_example+'/scripted_ops.yaml', 'w') as output:
    #     yaml.dump(traced_ops, output)


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
    parser.add_argument('--pruning_amount', type=float, default=0.3,
                        help='percentage of model to be pruned between 0 to 1')
    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    evaluate_model(args)
