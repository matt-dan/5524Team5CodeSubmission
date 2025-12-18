# 5524Team5CodeSubmission

## Google Colab Link:
[https://drive.google.com/drive/folders/1MH8K9tMeCvb1CLYaCCRPuX_JUuGoh8D2?usp=drive_link](https://colab.research.google.com/drive/1MlfNSj5E_J33FAcndicUizUB6hml6o8K?usp=sharing)
## Google Drive Link:
https://drive.google.com/drive/folders/1MH8K9tMeCvb1CLYaCCRPuX_JUuGoh8D2?usp=sharing

## RexOmni Installation Commands:
* pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128
* pip install -r requirements.txt
* pip install -v -e .

## Advanced Algorithm:
Our advanced algorithm was using RexOmni to generate new bounding boxes/labels for our training images and then using those new labels to train the yolov8n classifer.To run the RexOmni approach, use the google colab link and make sure that the shared google drive folder is within your google drive. Run the entire notebook to see how the bounding boxes and class labels are generated. Our best yolov8n model can be found within this github inside the "YOLOModel" directory. The provided predict.py will try to detect all of the weeds in the testing images. The mAP@50 score can be used to calculate the performance of our model and the viusalization python file will draw the bounding boxes on an image to show how it performs.

How to generate new labels using RexOmni model: Replace model_path, image_path, image_dir and label_dir with preferred paths.

Use predict.py to generate the bounding boxes for each image

Use calculate_score.py to calculate mAP score

## Test Images:
Images can be found at ./test/images
Labels can be found at ./test/partial_test_labels.csv
