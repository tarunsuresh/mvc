import cv2
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps


X,y=fetch_openml("mnist_784",version=1,return_X_y=True)
#print(pd.Series(y).value_counts())
classes=["0","1","2","3","4","5","6","7","8","9"]
nclasses=len(classes)

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=9,train_size=7500,test_size=2500)
X_train_scaled=X_train/255.0
X_test_scaled=X_test/255.0

Classifier=LogisticRegression(solver="saga",multi_class="multinomial").fit(X_train_scaled,y_train)

def get_prediction(img):
    im_pil=Image.open(img)
    image_gray=im_pil.convert("L")
    image_gray_resize=image_gray.resize((28,28),Image.ANTIALIAS)
    image_gray_resize_inverted=PIL.ImageOps.invert(image_gray_resize)
    pixel_filter=20
    min_pixel=np.percentile(image_gray_resize_inverted,pixel_filter)
    image_gray_resize_inverted_scalled=np.clip(image_gray_resize_inverted-min_pixel,0,255)
    maximum_pixal=np.max(image_gray_resize_inverted)
    image_gray_resize_inverted_scalled=np.asarray(image_gray_resize_inverted_scalled)/maximum_pixal
    test_sample=np.array(image_gray_resize_inverted_scalled).reshape(1,784)
    test_pred=Classifier(test_sample)

    #accuracy=accuracy_score(y_test,y_pred)
    return test_pred[0]

    
