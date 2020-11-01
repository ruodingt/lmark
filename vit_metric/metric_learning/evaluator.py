from collections import defaultdict, OrderedDict, Counter
from typing import Dict, Any, Tuple, List, Union
import numpy as np
import torch
from detectron2.evaluation import DatasetEvaluator
import re
import pprint


class Outputs:
    def __init__(self, outputs_dict):
        self._losses = {}
        self._fields = {}
        for a, b in outputs_dict.items():
            if re.search('loss', a):
                self._losses[a] = b.detach().cpu().numpy()
            else:
                self._fields[a] = b.detach().cpu()

    def __getitem__(self, item):
        _f = {}
        for a, b in self._fields.items():
            _f[a] = b[item].view(1, -1)
        return _f

    def get_loss_dict(self):
        return self._losses


class LossDict:
    def __init__(self):
        self._losses = defaultdict(list)

    def append_loss(self, loss_dict):
        for a, b in loss_dict.items():
            self._losses[a].append(b)

    def get_final_mean_loss(self):
        _losses_final = {}
        for a, ls in self._losses.items():
            _losses_final[a] = np.mean(ls)

        return _losses_final


class ClsDatasetEvaluator(DatasetEvaluator):
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """
    def __init__(self):
        self.val_loss_dict = LossDict()
        self.prob_scores = []
        self.pred_labels = []
        self.targets = []

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        self.val_loss_dict = LossDict()
        self.prob_scores = []
        self.pred_labels = []
        self.targets = []

    def process(self, inputs: Union[List, Dict], outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        For CLS problem, the out put looks like:
        {
        'label': label,
        'similarity': similarity,
        'classwise_cos_similarities': classwise_cos_similarities,
        'loss_arc_margin': loss
        }

        Args:
            inputs (list|dict): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        if isinstance(outputs, dict):
            outputs = Outputs(outputs)
            self.val_loss_dict.append_loss(outputs.get_loss_dict())
        else:
            raise NotImplementedError()

        for inp, out in zip(inputs, outputs):
            self.prob_scores.append(out['similarity'])
            self.pred_labels.append(out['label'])
            self.targets.append(inp['label'])
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """

        val_loss = self.val_loss_dict.get_final_mean_loss()
        prob_scores = torch.cat(self.prob_scores).numpy().squeeze()
        pred_labels = torch.cat(self.pred_labels).numpy().squeeze()
        targets = torch.stack(self.targets).numpy()

        acc_m = (pred_labels == targets).mean()
        y_true = {idx: target if target >= 0 else None for idx, target in enumerate(targets)}
        y_pred_m = {idx: (pred_cls, conf) for idx, (pred_cls, conf) in enumerate(zip(pred_labels, prob_scores))}
        gap_m = global_average_precision_score(y_true, y_pred_m)

        val_out = {'acc_m': acc_m, 'gap_m': gap_m}
        val_out.update(val_loss)

        val_out_ordered = OrderedDict()
        val_out_ordered['eval'] = val_out
        pprint.pprint(val_out)
        cc_pred = Counter(pred_labels.tolist())

        print(f"Most common preds:\n {cc_pred.most_common(10)}")

        cc_gt = Counter(targets.tolist())
        print(f"Most common gts:\n {cc_gt.most_common(10)}")

        return val_out_ordered



# def val_epoch(model, valid_loader, criterion, get_output=False):
#
#     model.eval()
#     val_loss = []
#     PRODS_M = []
#     PREDS_M = []
#     TARGETS = []
#
#     with torch.no_grad():
#         for (data, target) in tqdm(valid_loader):
#             data, target = data.cuda(), target.cuda()
#
#             logits_m = model(data)
#
#             lmax_m = logits_m.max(1)
#             probs_m = lmax_m.values
#             preds_m = lmax_m.indices
#
#             PRODS_M.append(probs_m.detach().cpu())
#             PREDS_M.append(preds_m.detach().cpu())
#             TARGETS.append(target.detach().cpu())
#
#             loss = criterion(logits_m, target)
#             val_loss.append(loss.detach().cpu().numpy())
#
#         val_loss = np.mean(val_loss)
#         PRODS_M = torch.cat(PRODS_M).numpy()
#         PREDS_M = torch.cat(PREDS_M).numpy()
#         TARGETS = torch.cat(TARGETS)
#
#     if get_output:
#         return LOGITS_M
#     else:
#         acc_m = (PREDS_M == TARGETS.numpy()).mean() * 100.
#         y_true = {idx: target if target >=0 else None for idx, target in enumerate(TARGETS)}
#         y_pred_m = {idx: (pred_cls, conf) for idx, (pred_cls, conf) in enumerate(zip(PREDS_M, PRODS_M))}
#         gap_m = global_average_precision_score(y_true, y_pred_m)
#         return val_loss, acc_m, gap_m


def global_average_precision_score(
        y_true: Dict[Any, Any],
        y_pred: Dict[Any, Tuple[Any, float]]
) -> float:
    """
    Compute Global Average Precision score (GAP)
    Parameters
    ----------
    y_true : Dict[Any, Any]
        Dictionary with query ids and true ids for query samples
    y_pred : Dict[Any, Tuple[Any, float]]
        Dictionary with query ids and predictions (predicted id, confidence
        level)
    Returns
    -------
    float
        GAP score
    """
    indexes = list(y_pred.keys())
    indexes.sort(
        key=lambda x: -y_pred[x][1],
    )
    queries_with_target = len([i for i in y_true.values() if i is not None])
    correct_predictions = 0
    total_score = 0.
    for i, k in enumerate(indexes, 1):
        relevance_of_prediction_i = 0
        if y_true[k] == y_pred[k][0]:
            correct_predictions += 1
            relevance_of_prediction_i = 1
        precision_at_rank_i = correct_predictions / i
        total_score += precision_at_rank_i * relevance_of_prediction_i

    return 1 / queries_with_target * total_score
