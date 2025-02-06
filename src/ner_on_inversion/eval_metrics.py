from collections import defaultdict


def extract_entities(tagged_sequence):
    """
    Extract entities from a tagged sequence in IOB format.
    Each entity is returned as a dictionary with keys:
      - 'tokens': list of tokens making up the entity
      - 'tag': the entity type
    """
    entities = []
    current_entity = None
    for token, tag, label in tagged_sequence:
        if label == 'B':
            if current_entity:
                entities.append(current_entity)
            current_entity = {'tokens': [token], 'tag': tag}
        elif label == 'I' and current_entity:
            current_entity['tokens'].append(token)
        else:
            if current_entity:
                entities.append(current_entity)
            current_entity = None
    if current_entity:
        entities.append(current_entity)
    return entities


def evaluate_ner(gold_standard, predictions):
    """
    Evaluate NER predictions against the gold standard on an entity-type basis.
    Both gold_standard and predictions should be lists of sequences (e.g., one per sentence)
    where each sequence is a list of tuples: (token, entity_type, label)
    """
    # Counters per entity type
    counts = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    # Overall counts (all entity types combined)
    overall_tp, overall_fp, overall_fn = 0, 0, 0

    # Process each sequence (e.g., each sentence)
    for gold_seq, pred_seq in zip(gold_standard, predictions):
        gold_entities = extract_entities(gold_seq)
        pred_entities = extract_entities(pred_seq)

        # To keep track of which gold entities have been matched in this sentence.
        matched_gold_indices = set()

        # Process each predicted entity
        for pred_entity in pred_entities:
            matched = False
            # Try to match the predicted entity with one of the gold entities.
            for j, gold_entity in enumerate(gold_entities):
                if j in matched_gold_indices:
                    continue  # Already matched this gold entity
                # Check for exact match in both tokens and entity type.
                if (tuple(pred_entity['tokens']) == tuple(gold_entity['tokens']) and
                        pred_entity['tag'] == gold_entity['tag']):
                    # If a match is found, count it as a true positive for that entity type.
                    counts[pred_entity['tag']]['tp'] += 1
                    overall_tp += 1
                    matched_gold_indices.add(j)
                    matched = True
                    break
            if not matched:
                # If no match was found, this prediction is a false positive.
                counts[pred_entity['tag']]['fp'] += 1
                overall_fp += 1

        # Any gold entity not matched is a false negative.
        for j, gold_entity in enumerate(gold_entities):
            if j not in matched_gold_indices:
                counts[gold_entity['tag']]['fn'] += 1
                overall_fn += 1

    # Now compute precision, recall, and F1 for each entity type.
    metrics = {}
    for entity_type, data in counts.items():
        tp = data['tp']
        fp = data['fp']
        fn = data['fn']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        metrics[entity_type] = {
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'TP': tp, 'FP': fp, 'FN': fn
        }

    # Optionally, also compute overall metrics
    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
    overall_f1 = (2 * overall_precision * overall_recall / (overall_precision + overall_recall)
                  if (overall_precision + overall_recall) > 0 else 0)
    metrics['Overall'] = {
        'Precision': overall_precision,
        'Recall': overall_recall,
        'F1-Score': overall_f1,
        'TP': overall_tp, 'FP': overall_fp, 'FN': overall_fn
    }

    return metrics
