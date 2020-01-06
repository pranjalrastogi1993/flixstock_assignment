# %% package imports
#basic imports
import os
import numpy as np
import cv2
import json
from keras.models import load_model
from matplotlib import pyplot as plt

#%%
#Change path names for running inference
image_path = 'images/1a8bd69a-9720-41a4-8719-8607d3f061d21535966638226-Peter-England-Men-Navy-Blue-Self-Design-Polo-Collar-T-shirt-141535966638022-1.jpg'
model_path = 'FashionNet_Submitted.h5'

#%%
#function to preprocess the image
def preprocImg(img):
    ti = np.float32(img)/255.0
    return ti

#%%
#function to get inference on an image
def inferOnImage(image,model,thresh=0.5):
        im_pp = preprocImg(image)
        im_pp = np.expand_dims(im_pp,axis=0)
        #predict via model
        pred = model.predict(im_pp)
        neckop = pred[0][0:8]
        sleeveop = pred[0][8:13]
        patternop = pred[0][13:24]
        return neckop,sleeveop,patternop

#%%
#load the json file with testData
testData_jsonpath = 'testData.json'
with open(testData_jsonpath) as jsonFile:
    testData = json.load(jsonFile)

#%%
#load the model to disk
themodel = load_model(model_path,compile=False)

#%%    
#function for inference on test data
def inferenceTestset(model,testData,imgfolpath,thresh=0.5):
    #check for empty vals
    if model and testData:
        #classsplit = [8,5,11]
        for t in testData:
            fname = t['filename']
            neck = t['neck']
            sleeve = t['sleeve_length']
            pattern = t['pattern']
            #update filepath
            new_fname = os.path.join(imgfolpath,fname)
            img = cv2.imread(new_fname,1)
            if img is None:
                continue
            neckop,sleeveop,patternop = inferOnImage(img,model,thresh)
            #display image and predictions
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.show()
            print('GROUND TRUTH: \nNeck: {0}\nSleeve: {1}\nPattern: {2}'.format(neck,sleeve,pattern))
            print('\nPREDICTED:')
            for n in range(len(neckop)):
                if neckop[n] >= thresh:
                    if n == len(neckop)-1:
                        print('Neck: {0}'.format('#N/A'))
                    else:
                        print('Neck: {0}'.format(n))
            for s in range(len(sleeveop)):
                if sleeveop[s] >= thresh:
                    if s == len(sleeveop)-1:
                        print('Sleeve: {0}'.format('#N/A'))
                    else:
                        print('Sleeve: {0}'.format(s))
            for p in range(len(patternop)):
                if patternop[p] >= thresh:
                    if p == len(patternop)-1:
                        print('Pattern: {0}'.format('#N/A'))
                    else:
                        print('Pattern: {0}'.format(p))
            #break

inferenceTestset(themodel,testData,imgfolpath='images',thresh=0.4)

#%%
#inference code
def runInference(image_path,model,thresh=0.5):
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        
        neckop,sleeveop,patternop = inferOnImage(img,model,thresh)
        
        #display image and predictions
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.show()
        print('\nPREDICTED:')
        for n in range(len(neckop)):
            if neckop[n] >= thresh:
                if n == len(neckop)-1:
                    print('Neck: {0}'.format('#N/A'))
                else:
                    print('Neck: {0}'.format(n))
        for s in range(len(sleeveop)):
            if sleeveop[s] >= thresh:
                if s == len(sleeveop)-1:
                    print('Sleeve: {0}'.format('#N/A'))
                else:
                    print('Sleeve: {0}'.format(s))
        for p in range(len(patternop)):
            if patternop[p] >= thresh:
                if p == len(patternop)-1:
                    print('Pattern: {0}'.format('#N/A'))
                else:
                    print('Pattern: {0}'.format(p))

runInference(image_path,themodel,thresh=0.4)
        