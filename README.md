# 5524Team5CodeSubmission

RexOmni Installation Commands:
pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -v -e .

How to generate new labels using RexOmni model:
Replace model_path, image_dir, label_dir, image_path with preferred paths.

---Advanced Algorithm---
How to generate new labels using RexOmni model: Replace model_path, image_dir and label_dir with preferred paths.

Use predict.py to generate the bounding boxes for each image

Use calculate_score.py to calculate mAP score

---Test Images---
images can be found at ./test/images
labels can be found at ./test/partial_test_labels.csv

autumn 2025 comp vision code for team 5

---
