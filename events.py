from utilities import draw_debug_images,draw_mask, COCOResults
import torch
from trains import Task


def evaluation_started(engine):
    engine.state.eval_size = 0
    engine.state.running_corrects = 0
    engine.state.failures = []
    engine.state.confusion_matrix = torch.zeros(size=(2, 2))


def eval_iteration_completed(engine, writer, task_args, report_iter=None):
    curr_iteration = report_iter if report_iter is not None else engine.state.iteration
    images, targets, results, preds, iter_corrects = engine.state.output
    engine.state.eval_size += 1
    engine.state.running_corrects += sum(iter_corrects)

    for image, target, pred, correct in zip(images, targets, preds, iter_corrects):
        engine.state.confusion_matrix[int(len(target['labels']) > 0), int(len(pred) > 0)] += 1
        if len(engine.state.failures) < 20:
            if not correct:
                debug_images = draw_debug_images([image], [target], results, task_args.test_score_thr)
                engine.state.failures.append(debug_images[0])

    if engine.state.iteration % task_args.log_interval == 0:
        print("Evaluation: Iteration: {}".format(engine.state.iteration))
    if engine.state.iteration % task_args.debug_images_interval == 0:
        for n, debug_image in enumerate(draw_debug_images(images, targets, results, task_args.test_score_thr)):
            writer.add_image("evaluation/image_{}_{}".format(engine.state.iteration, n),
                             debug_image, curr_iteration, dataformats='HWC')
            if 'masks' in targets[n]:
                writer.add_image("evaluation/image_{}_{}_mask".format(engine.state.iteration, n),
                                 draw_mask(targets[n]), curr_iteration, dataformats='HW')
                curr_image_id = int(targets[n]['image_id'])
                writer.add_image("evaluation/image_{}_{}_predicted_mask".format(engine.state.iteration, n),
                                 draw_mask(results[curr_image_id]).squeeze(), curr_iteration,
                                 dataformats='HW')
    images = targets = results = engine.state.output = None


def evaluation_completed(engine, writer, task_args, report_iter=None):
    # gather the stats from all processes TODO: will fail if there is only on process
    engine.state.coco_evaluator.synchronize_between_processes()
    # accumulate predictions from all images
    engine.state.coco_evaluator.accumulate()
    engine.state.coco_evaluator.summarize()

    curr_iteration = report_iter if report_iter is not None else engine.state.iteration
    frame_accuracy = float(engine.state.running_corrects) / (engine.state.eval_size * task_args.batch_size)
    writer.add_scalar('Frame Performance/accuracy_labels', frame_accuracy, curr_iteration)

    cm_labels = ['no_target', 'target']
    cm = engine.state.confusion_matrix.numpy()
    Task.current_task().get_logger().report_confusion_matrix('Val summary', 'Confusion matrix', cm,
                                                             iteration=curr_iteration, xaxis='Prediction',
                                                             yaxis='GT',
                                                             xlabels=cm_labels, ylabels=cm_labels)
    true_positive = cm[1, 1]
    true_negative = cm[0, 0]
    false_positive = cm[0, 1]
    total_positive = cm[1, :].sum()
    total_negatives = cm[0, :].sum()
    total_predicted_positive = cm[:, 1].sum()
    writer.add_scalar('Frame Performance/precision', true_positive / total_predicted_positive, curr_iteration)
    writer.add_scalar('Frame Performance/recall', true_positive / total_positive, curr_iteration)
    writer.add_scalar('Frame Performance/FPR', false_positive / total_negatives, curr_iteration)
    writer.add_scalar('Frame Performance/accuracy',
                      (true_positive + true_negative) / (total_positive + total_negatives), curr_iteration)

    results_summary = COCOResults(engine.state.coco_evaluator, engine.state.label_enum)
    full_results = results_summary.get_results()
    for iou_type in engine.state.coco_evaluator.iou_types:
        for key, val in full_results[iou_type]['Average Precision'].items():
            writer.add_scalar("Average Precision-{}/{}".format(iou_type, key), val, curr_iteration)
        for key, val in full_results[iou_type]['Average Recall'].items():
            writer.add_scalar("Average Recall-{}/{}".format(iou_type, key), val, curr_iteration)
        for key, val in full_results[iou_type]['AP50'].items():
            writer.add_scalar("AP50 per class-{}/{}".format(iou_type, key), val, curr_iteration)
        Task.current_task().get_logger().report_line_plot(
            title="Precision-Recall - %s @IOU:%s" % (iou_type, results_summary.iou_thresh),
            series=full_results[iou_type]['Precision-Recall Curve'],
            iteration=curr_iteration,
            xaxis='Recall',
            yaxis='Precision')
        Task.current_task().get_logger().report_line_plot(
            title="Conf-F1-score - %s @IOU:%s" % (iou_type, results_summary.iou_thresh),
            series=full_results[iou_type]['Conf-F1-score Curve'],
            iteration=curr_iteration,
            xaxis='Conf. Threshold',
            yaxis='F1-Score')

    if len(engine.state.failures) > 0:
        for n, failure in enumerate(engine.state.failures):
            writer.add_image("failures/image_{}".format(n),
                             failure, curr_iteration, dataformats='HWC')
