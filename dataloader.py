#==========DATA-LOADER FOR MACHINE LEARNING FRAMEWORKS==========#
#==========MADE WITH LOVE BY RUMEET SINGH==========# 

import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split

class DataLoader:

    def __init__(self,DIR,CATEGORIES):
        self.DIR = DIR
        self.CATEGORIES = CATEGORIES

    def create_data(self,cmap,test_size=0.25,random_state=None,normalize=False,size_x=50,size_y=50):
        training_data = []
        X = []
        y = []

        for category in self.CATEGORIES:
            path = os.path.join(self.DIR,category)
            class_num = self.CATEGORIES.index(category)

            for img in os.listdir(path):
                if cmap=='rgb':
                    try:
                        brg_img = cv2.imread(os.path.join(path,img))
                        b,g,r = cv2.split(brg_img)       # get b,g,r
                        img_array = cv2.merge([r,g,b])     # switch it to rgb
                        img_array = cv2.resize(img_array,(size_x,size_y))
                        if normalize:
                            img_array = img_array/255
                        training_data.append([img_array,class_num])
                    except Exception as e:
                        pass 
                elif cmap=='gray':
                    try:
                        img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                        if normalize:
                            img_array = img_array/255
                        training_data.append([img_array,class_num])
                    except Exception as e:
                        pass 

        for images,labels in training_data:
            X.append(images)
            y.append(labels)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,random_state=random_state)
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test) 

        if cmap=='gray':
            X_train = np.expand_dims(X_train, -1)
            X_test = np.expand_dims(X_test, -1)

        return X_train, X_test, y_train, y_test
