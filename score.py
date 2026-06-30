import json
import argparse

def compute_iou(interval_1, interval_2):
    start_i, end_i = interval_1[0], interval_1[1]
    start, end = interval_2[0], interval_2[1]
    intersection = max(0, min(end, end_i) - max(start, start_i))
    union = min(max(end, end_i) - min(start, start_i), end-start + end_i-start_i)
    iou = float(intersection) / (union + 1e-8)
    return iou

def evaluate_iou(predictions, labels, thresholds=[0.5, 0.7]):
    results = {threshold: {"correct": 0, "total": 0} for threshold in thresholds}
    
    # Count non-empty intervals in labels
    valid_labels = sum(1 for label in labels if len(label) == 2)
    
    for pred, label in zip(predictions, labels):
        # Skip if either prediction or label is empty
        if len(pred) != 2 or len(label) != 2:
            continue
            
        iou_score = compute_iou(pred, label)
        
        # Check each threshold
        for threshold in thresholds:
            results[threshold]["total"] = valid_labels
            if iou_score >= threshold:
                results[threshold]["correct"] += 1
    
    # Calculate accuracy for each threshold
    for threshold in thresholds:
        correct = results[threshold]["correct"]
        total = results[threshold]["total"]
        accuracy = correct / total * 100 if total > 0 else 0
        print(f"IoU@{threshold}: {accuracy:.4f} ({correct}/{total})")
    
    return results

# from rouge import rouge
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


def rouge_scorer(hyp_list, ref_list):
    # Initialize the scorer
    # scorer = rouge.rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Calculate scores
    scores = [scorer.score(ref, hyp) for hyp, ref in zip(hyp_list, ref_list)]

    sums = {
        'rouge1': {'precision': 0, 'recall': 0, 'fmeasure': 0},
        'rouge2': {'precision': 0, 'recall': 0, 'fmeasure': 0},
        'rougeL': {'precision': 0, 'recall': 0, 'fmeasure': 0}
    }

    # Sum up all the scores
    for score in scores:
        for key in sums.keys():
            sums[key]['precision'] += score[key].precision
            sums[key]['recall'] += score[key].recall
            sums[key]['fmeasure'] += score[key].fmeasure

    # Calculate averages
    num_pairs = len(hyp_list)
    averages = {
        key: {
            'precision': sums[key]['precision'] / num_pairs,
            'recall': sums[key]['recall'] / num_pairs,
            'fmeasure': sums[key]['fmeasure'] / num_pairs
        }
        for key in sums.keys()
    }

    # Output the average scores
    print("Average ROUGE scores:")
    for key in averages.keys():
        print(f"{key}:")
        for metric in ['precision', 'recall', 'fmeasure']:
            print(f"  {metric}: {averages[key][metric]:.4f}")
            
    return averages[key][metric] # ROUGE_L fmeasure

def add_arguments(parser):
    
    parser.add_argument("--label_path", type=str, default="/media/allen/documents/research/LENS/data/splits/test.json")
    parser.add_argument("--moment_path", type=str, default="/media/allen/documents/research/LENS/checkpoints_best/test_moment_retrieval_BEST.json")
    parser.add_argument("--awesome_path", type=str, default="/media/allen/documents/research/LENS/checkpoints_best/test_awesome_BEST.json")


def main(args):

    with open(args.label_path) as f:
        label_json = json.load(f)

    with open(args.awesome_path) as f:
        pred_json = json.load(f)

    with open(args.moment_path) as f:
        pred_time_json = json.load(f)

    abs_label_dict, time_label_list = {}, []
    for prompt in label_json:
        for k in label_json[prompt]:
            if k not in abs_label_dict:
                abs_label_dict[k] = []
            time_label_list.append(label_json[prompt][k]['bounds'])
            abs_label_dict[k].append(label_json[prompt][k]['summary'])

    time_pred_list = []
    for prompt in pred_time_json:
        for k in pred_time_json[prompt]:
            time_pred_list.append(pred_time_json[prompt][k]['bounds'])

            
    abs_label_list = []
    for i in abs_label_dict:
        text = ' '.join(abs_label_dict[i])
        filtered_text = text.replace("no summary provided", "").strip()
        abs_label_list.append(filtered_text)
        
    abs_pred_list = []
    for i in pred_json:
        text = ' '.join(pred_json[i]['summary'])
        filtered_text = text.replace("no summary provided", "").strip()
        abs_pred_list.append(filtered_text)

    evaluate_iou(time_pred_list, time_label_list)
    rouge_scorer(abs_pred_list, abs_label_list)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    main(args)