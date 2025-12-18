from PIL import Image, ImageDraw, ImageFont
import pandas as pd

predictions = pd.read_csv("predictions.csv", dtype={"prediction_string" : str})
image = "test/images/20190613_6062W_CM_36.jpg"
out_path = "visualization.jpg"

row = predictions[predictions["image_id"] == "20190613_6062W_CM_36"]
extracted_prediction = row["prediction_string"].values[0]




img = Image.open(image).convert("RGB")
draw = ImageDraw.Draw(img)

data = [float(x) for x in extracted_prediction.split()]
img_w, img_h = img.size
for i in range(0, len(data), 6):
    label, confidence, x_center, y_center, width, height = data[i : i + 6]
    left = round((x_center - width / 2) * img_w)
    right = round((x_center + width /2 ) * img_w)
    top = round((y_center - height / 2) * img_h)
    bottom = round((y_center + height / 2) * img_h)

    left = max(0, left)
    right = min(img_w - 1, right)
    top = max(0, top)
    bottom = min(img_h -1, bottom)

    draw.rectangle([left, top, right, bottom], outline="red", width = 3)

img.save(out_path)


