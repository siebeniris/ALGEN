import numpy as np
import evaluate


def eval_classification(references, logits_output, classification="binary"):
    # logits, labels = eval_pred
    predictions = np.argmax(logits_output, axis=-1)

    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    if classification == "multiclass":
        auc_metric = evaluate.load("roc_auc", classification)
        auc_results = auc_metric.compute(references=references, prediction_scores=logits_output,
                                        multi_class="ovo")

    else:
        auc_metric = evaluate.load("roc_auc")
        auc_results = auc_metric.compute(references=references, prediction_scores=logits_output)

    auc_score = round(auc_results["roc_auc"], 4)*100

    f1_result = f1_metric.compute(predictions=predictions, references=references, average="macro")
    f1_score = round(f1_result["f1"], 4)*100

    acc_result = accuracy_metric.compute(references=references, predictions=predictions)
    acc_score = round(acc_result["accuracy"],4)*100

    return acc_score, f1_score, auc_score

