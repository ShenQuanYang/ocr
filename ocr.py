import cv2
import pytesseract
from pytesseract import Output
import re
import numpy as np
import pandas as pd
import matplotlib as plt

# 加载图像
image = cv2.imread('example.png')

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用二值化
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# 识别图像中的文本
text = pytesseract.image_to_string(thresh, lang='eng')

# 打印识别出的文本
print("识别的文本:")
print(text)

# 获取每个字符的位置信息并绘制边界框
data = pytesseract.image_to_data(thresh, output_type=Output.DICT)
n_boxes = len(data['level'])
for i in range(n_boxes):
    (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 显示带有边界框的图像
cv2.imshow('image', image)
cv2.waitKey(0)

# 简单的后处理示例：去除所有非字母数字字符
cleaned_text = re.sub(r'\W+', ' ', text)
print("清理后的文本:")
print(cleaned_text)
