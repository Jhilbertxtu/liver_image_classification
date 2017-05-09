"""liver datasets utilities

"""

from __future__ import absolute_import, print_function

import os
import xlrd
import numpy as np
from PIL import Image
from sklearn import preprocessing


def load_image(img_path):
    img = Image.open(img_path)
    return img

def resize_image(in_image, new_width, new_height, out_image=None,
                 resize_mode=Image.ANTIALIAS):
    img = in_image.resize((new_width, new_height), resize_mode)
    if out_image:
        img.save(out_image)
    return img

def pil_to_nparray(pil_image):
    pil_image.load()
    return np.asarray(pil_image, dtype="float32")

def load_data(dirname="liver_pierce_images", sheet_name="G0", resize=(227, 227), to_gray=False, one_hot=False, verbose=False):
    xlsx_path = os.path.join(dirname, "annotation.xlsx")
    data = xlrd.open_workbook(xlsx_path)
    table = data.sheet_by_name(sheet_name)
    img_paths = table.col_values(0)
    img_paths.pop()
    img_labels = table.col_values(1)
    img_labels.pop()

    images = []
    for img_path in img_paths:
        full_path = os.path.join(dirname, sheet_name, img_path)
        if verbose:
            print(full_path)
        img = load_image(full_path)
        if to_gray:
            img = img.convert('L')
        img = resize_image(img, resize[0], resize[1])
        np_img = pil_to_nparray(img)
        images.append(np_img)
    images = np.array(images)

    labelenc = preprocessing.LabelEncoder()
    labelenc.fit(img_labels)
    labels = labelenc.transform(img_labels)
    labels = np.array(labels).reshape(-1, 1)

    onehotenc = None
    if one_hot:
        onehotenc = preprocessing.OneHotEncoder()
        onehotenc.fit(labels)
        labels = onehotenc.transform(labels)

    return images, labels, labelenc, onehotenc, img_labels

if __name__ == '__main__':
    X, Y, labelenc, onehotenc, img_labels = load_data(one_hot=True, to_gray=True)
    print("X shapes and Y shapes:")
    print(X.shape, Y.shape)