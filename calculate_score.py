import numpy as np
import pandas as pd
import ultralytics
def parse_prediction_string(s, with_conf=False):
    """
    Parses the prediction string into a list of boxes.
    GT Format: class cx cy w h
    Pred Format: class conf cx cy w h
    """
    if pd.isna(s) or s == "": 
        return []
    
    parts = s.strip().split()
    boxes = []
    
    step = 6 if with_conf else 5
    
    for i in range(0, len(parts), step):
        if i + step > len(parts): 
            break
            
        try:
            cls = int(parts[i])
            if with_conf:
                # Format: class conf cx cy w h
                conf = float(parts[i+1])
                cx = float(parts[i+2])
                cy = float(parts[i+3])
                w = float(parts[i+4])
                h = float(parts[i+5])
                boxes.append([cls, conf, cx, cy, w, h])
            else:
                # Format: class cx cy w h
                cx = float(parts[i+1])
                cy = float(parts[i+2])
                w = float(parts[i+3])
                h = float(parts[i+4])
                boxes.append([cls, cx, cy, w, h])
        except ValueError:
            continue # Skip malformed parts
            
    return boxes

def compute_iou(box1, box2):
    """Calculates Intersection over Union (IoU) for two boxes [cx, cy, w, h]."""
    cx1, cy1, w1, h1 = box1
    cx2, cy2, w2, h2 = box2
    
    # Convert Center (cx, cy, w, h) to Corner (x1, y1, x2, y2)
    x1_min, y1_min = cx1 - w1/2, cy1 - h1/2
    x1_max, y1_max = cx1 + w1/2, cy1 + h1/2
    x2_min, y2_min = cx2 - w2/2, cy2 - h2/2
    x2_max, y2_max = cx2 + w2/2, cy2 + h2/2
    
    # Intersection area
    inter_w = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_h = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    inter_area = inter_w * inter_h
    
    # Union area
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0
        
    return inter_area / union_area

def calculate_map(solution_path, submission_path):
    # Load data
    gt_df = pd.read_csv(solution_path)
    pred_df = pd.read_csv(submission_path)
    
    # Merge on image_id to ensure we only evaluate shared images
    merged = pd.merge(gt_df, pred_df, on='image_id', suffixes=('_gt', '_pred'))
    print(f"Evaluating on {len(merged)} images...")

    classes = set()
    for s in merged['prediction_string_gt']:
        boxes = parse_prediction_string(s, with_conf=False)
        for b in boxes:
            classes.add(int(b[0]))
    
    classes = sorted(list(classes))
    print(f"Classes found: {classes}")
    
    aps = []

    for cls in classes:
        # Collect all Predictions and Ground Truths for this specific class
        preds = []  # List of [conf, image_id, box]
        gts = {}    # Dict of image_id -> list of {'box': box, 'used': False}
        
        # We must track how many GT boxes exist for this class across ALL images
        total_gt_boxes_for_class = 0
        
        for idx, row in merged.iterrows():
            img_id = row['image_id']
            
            # Parse GT for this image
            gt_boxes_raw = parse_prediction_string(row['prediction_string_gt'], False)
            class_gt_boxes = [b[1:] for b in gt_boxes_raw if int(b[0]) == cls]
            
            if class_gt_boxes:
                if img_id not in gts: gts[img_id] = []
                for box in class_gt_boxes:
                    gts[img_id].append({'box': box, 'used': False})
                total_gt_boxes_for_class += len(class_gt_boxes)
            
            # Parse Preds for this image
            pred_boxes_raw = parse_prediction_string(row['prediction_string_pred'], True)
            class_pred_boxes = [b[1:] for b in pred_boxes_raw if int(b[0]) == cls]
            
            for p_box in class_pred_boxes:
                # p_box is [conf, cx, cy, w, h]
                conf = p_box[0]
                box_coords = p_box[1:]
                preds.append([conf, img_id, box_coords])

        if total_gt_boxes_for_class == 0:
            print(f"Class {cls}: No Ground Truth boxes found. AP = 0.0")
            aps.append(0.0)
            continue

        # Sort predictions by confidence (Highest first)
        preds.sort(key=lambda x: x[0], reverse=True)
        
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        
        # Match Predictions to GT
        for i, (conf, img_id, pred_box) in enumerate(preds):
            best_iou = 0
            best_gt_idx = -1
            
            if img_id in gts:
                for idx, gt_obj in enumerate(gts[img_id]):
                    iou = compute_iou(pred_box, gt_obj['box'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx
            
            # IoU Threshold 0.5
            if best_iou >= 0.5:
                if not gts[img_id][best_gt_idx]['used']:
                    tp[i] = 1
                    gts[img_id][best_gt_idx]['used'] = True
                else:
                    fp[i] = 1 # Duplicate detection (already matched)
            else:
                fp[i] = 1 # Poor overlap or hallucination

        # Compute AP using All-Points Interpolation
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / total_gt_boxes_for_class
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Smooth precision curve (make it monotonically decreasing)
        # Add sentinel points for integration (0 and 1)
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))
        
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
            
        # Calculate Area Under Curve
        i_list = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i_list + 1] - mrec[i_list]) * mpre[i_list + 1])
        
        aps.append(ap)
        print(f"Class {cls} AP: {ap:.4f}")

    if not aps:
        print("No classes evaluated.")
    else:
        print(f"\nFinal mAP@0.5: {np.mean(aps):.4f}")

# --- UPDATE FILENAMES HERE ---
calculate_map('test/partial_test_labels.csv', 'predictions.csv')

