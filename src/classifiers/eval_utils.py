import numpy as np
import evaluate


def eval_classification(references, output, num_labels, classification):
    # logits, labels = eval_pred
    if num_labels > 2:
        predictions = np.argmax(output, axis=-1)
        print(predictions)
    else:
        output_array = np.array(output)
        predictions = (output_array > 0.5).astype(int)
        print(predictions)


    if classification == "multiclass":
        print("multiclass auc metric")
        auc_metric = evaluate.load("roc_auc", classification)
        auc_results = auc_metric.compute(references=references, prediction_scores=output,
                                         multi_class="ovo")
    else:
        print("binary class auc metric")
        positive_class_probs = output[:, 1].numpy()
        references = np.array(references)
        auc_metric = evaluate.load("roc_auc")
        auc_results = auc_metric.compute(references=references, prediction_scores=positive_class_probs)

    auc_score = round(auc_results["roc_auc"], 4) * 100

    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    f1_result = f1_metric.compute(predictions=predictions, references=references,
                                  average="macro" if classification == "multiclass" else "binary")
    f1_score = round(f1_result["f1"], 4) * 100

    acc_result = accuracy_metric.compute(references=references, predictions=predictions)
    acc_score = round(acc_result["accuracy"], 4) * 100

    return acc_score, f1_score, auc_score
