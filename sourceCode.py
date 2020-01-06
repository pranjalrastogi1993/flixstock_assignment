# %% package imports
#basic imports
import os
import numpy as np
import cv2
import json
import copy
import random
import albumentations as AL
import itertools
import tensorflow as tf
#keras imports
import keras
from keras import backend as K
from keras.layers import concatenate
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, AvgPool2D
from keras.layers import Activation, Flatten, Dense, Dropout, Lambda, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import matplotlib.pyplot as plt
from collections import Counter
from keras.preprocessing import image

#%%
#global varibales decleration
image_folpath = "images"
attrib_filepath = "attributes.json"

#%%
#function to load the dataset
def loadDS(attrib_fp):
    #load file
    with open(attrib_fp) as jsonFile:
        data = json.load(jsonFile)
    #remove records with 3 #N/A
    #also get attribute values range
    cleanData = []
    neckVals = []
    sleeveVals = []
    patternVals = []
    for record in data:
        neck = record['neck']
        sleeve = record['sleeve_length']
        pattern = record['pattern']
        
        if neck not in neckVals:
            neckVals.append(neck)
        if sleeve not in sleeveVals:
            sleeveVals.append(sleeve)
        if pattern not in patternVals:
            patternVals.append(pattern)
        
        count = 0
        if type(neck) == str:
            count += 1
        if type(sleeve) == str:
            count += 1
        if type(pattern) == str:
            count += 1
        
        if count == 3:
            continue
        else:
            cleanData.append(record)
    print("No. of valid records: {}".format(len(cleanData)))
    return cleanData,{'neck':neckVals,'sleeve':sleeveVals,'pattern':patternVals}
        
    
cleanDS,attrib_values = loadDS(attrib_filepath)

#%%
#function to get frequency distribution of attributes
def getFreqDist(dataset,attribvals,printit=False):
    #obtain attrib wise values
    neckvals = attribvals['neck']
    sleevevals = attribvals['sleeve']
    patternvals = attribvals['pattern']
    #build containers
    ndict = {}
    for n in neckvals:
        ndict[n] = 0
    sdict = {}
    for s in sleevevals:
        sdict[s] = 0
    pdict = {}
    for p in patternvals:
        pdict[p] = 0
    #populate containers
    for d in dataset:
        neck = d['neck']
        sleeve = d['sleeve_length']
        pattern = d['pattern']
        
        ndict[neck] += 1
        sdict[sleeve] += 1
        pdict[pattern] += 1
    
    if printit:
        print('Neck freq: \n',ndict)
        print('Sleeve freq: \n',sdict)
        print('Pattern freq: \n',pdict)
    return {'neck':ndict,'sleeve':sdict,'pattern':pdict}

ditributions = getFreqDist(cleanDS,attrib_values)

#%%
#function to split the dataset
def splitDS(dataset,attribvals,split1=0.8,split2=0.2):
    #verify the split
    addedratios = round(split1+split2,10)
    if  addedratios != 1.0:
        print('The split ratios do not add up to {0} instead of 1.0'.format(addedratios))
        return None
    #init the containers
    traindata=[]
    validdata=[]
    #get the freq distrib
    freqdis = getFreqDist(dataset,attribvals)
    nfreq = freqdis['neck']
    sfreq = freqdis['sleeve']
    pfreq = freqdis['pattern']
    freqdis_blank = getFreqDist([],attribvals)
    nfreq_count = freqdis_blank['neck']
    sfreq_count = freqdis_blank['sleeve']
    pfreq_count = freqdis_blank['pattern']
    #iterate over dataset
    np.random.shuffle(dataset)
    for d in dataset:
        neck = d['neck']
        sleeve = d['sleeve_length']
        pattern = d['pattern']
        if nfreq_count[neck] < nfreq[neck]*split2 or sfreq_count[sleeve] < sfreq[sleeve]*split2 or pfreq_count[pattern] < pfreq[pattern]*split2:
            validdata.append(d)
            nfreq_count[neck] += 1
            sfreq_count[sleeve] += 1
            pfreq_count[pattern] += 1
        else:
            traindata.append(d)
    return traindata,validdata

#%%
splitOne,testData = splitDS(cleanDS,attrib_values,split1=0.9,split2=0.1)
trainData,validData = splitDS(splitOne,attrib_values,split1=0.75,split2=0.25)    
#getFreqDist(trainData,attrib_values)

#%%
#create a json dump of test data
with open('testData.json', 'w') as f:
    json.dump(testData, f)
    
#%%
#setup the augmentation pipeline
augpipe = AL.Compose([
        AL.HorizontalFlip(),
        AL.RGBShift(),
        AL.RandomBrightness(),
        AL.RandomContrast(),
        AL.GaussNoise()])

#%%
#function to preprocess the image
def preprocImg(img):
    ti = np.float32(img)/255.0
    return ti  

#%%
#function to build one hot encoded multilabel vector
def multilabelOHE(record,attribvals):
    #extract attribute values
    neckvals = attribvals['neck']
    sleevevals = attribvals['sleeve']
    patternvals = attribvals['pattern']
    #extract record info
    neck = record['neck']
    sleeve = record['sleeve_length']
    pattern = record['pattern']
    #prep vectors
    neckvec = [0]*len(neckvals)
    sleevevec = [0]*len(sleevevals)
    patternvec = [0]*len(patternvals)
    
    if type(neck) == str:
        neckvec[-1] = 1
    else:
        neckvec[neck] = 1
        
    if type(sleeve) == str:
        sleevevec[-1] = 1
    else:
        sleevevec[sleeve] = 1
    
    if type(pattern) == str:
        patternvec[-1] = 1
    else:
        patternvec[pattern] = 1
    
    return neckvec+sleevevec+patternvec    

#%%
#function to build the dataset
#while countering data imbalance using selective augmentations
def buildDataset(dataset,attribvals,imgfolpath,augpipeline=None,copyOnDisk=False):
    #get dataset count
    dssize = len(dataset)
    #obtain attrib wise values
    neckvals = attribvals['neck']
    sleevevals = attribvals['sleeve']
    patternvals = attribvals['pattern']
    #get mean per attribute
    nmean = int(dssize/len(neckvals))
    smean = int(dssize/len(sleevevals))
    pmean = int(dssize/len(patternvals))
    #get attrib wise distrib
    attribdist = getFreqDist(dataset,attribvals)
    neckdis = attribdist['neck']
    sleevedis = attribdist['sleeve']
    patterndis = attribdist['pattern']
    #start building
    imgs = []
    labels = []
    for d in dataset:
        #extract attrib vals to be encoded
        fname = d['filename']
        neck = d['neck']
        sleeve = d['sleeve_length']
        pattern = d['pattern']
        #update img path
        new_fname = os.path.join(imgfolpath,fname)
        #load the image
        im = cv2.imread(new_fname,1)
        if im is None:
            continue
        im_pp = preprocImg(im)
        imgs.append(im_pp)
        #get one hot encoding
        ohe = multilabelOHE(d,attribvals)
        labels.append(ohe)
        #add augmentations if necessary
        if augpipeline:
            if neckdis[neck] <= nmean or sleevedis[sleeve] <= smean or patterndis[pattern] <= pmean:
                aug_i = augpipeline(image=im)
                augImg = aug_i['image']
                aug_pp = preprocImg(augImg)
                imgs.append(aug_pp)
                labels.append(ohe)
    #return imgs,labels
    return np.stack(imgs),np.stack(labels)

#%%
#get the training and validation data
t_imgs,t_labels = buildDataset(trainData,attrib_values,image_folpath,augpipeline=augpipe)

v_imgs,v_labels = buildDataset(validData,attrib_values,image_folpath,augpipeline=None)

# %% shuffle two numpy arrays in unision
def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

#%%
#batch generator function
def getBatch(datasplit,scoresplit,batch_size=64):
    incount = datasplit.shape[0]
    if incount < 1:
        print("No entries in the given split.")
    elif batch_size > incount:
        print("Batch size is larger than dataset size.")
    else:
        pointer=0
        while True:
            if pointer==0:
                shuffle_in_unison(datasplit,scoresplit)
            d = datasplit[pointer:pointer+batch_size]
            s = scoresplit[pointer:pointer+batch_size]
            
            pointer = pointer + batch_size
            if pointer >= incount:
                pointer = 0
            yield d,s

#%%
trainGen = getBatch(t_imgs,t_labels,batch_size=64)
sampleTrainBatch = next(trainGen)
inputshape = sampleTrainBatch[0].shape[1:]

#%%
from keras.applications import MobileNet
#function that returns a model
def getModel(input_shape,showsummary=False):
    #model architecture
    theinput = Input(shape=input_shape)
    themodel = MobileNet(input_tensor=theinput,include_top=False,weights=None)
    #freeze/unfreeze the network
    for layer in themodel.layers:
        layer.trainable = True
    #custom layers
    fnet = GlobalAveragePooling2D()(themodel.output)
    fnet = Dense(1024,activation='relu')(fnet)
    fnet = BatchNormalization()(fnet)
    fnet = Dropout(0.5)(fnet)
    fnet = Dense(512)(fnet)
    fnet = Dropout(0.33)(fnet)
    output = Dense(24,activation='sigmoid',name='finalOutput')(fnet)
    fmodel = Model([theinput],[output],name='FashionNet')
    
    if showsummary:
        #model summary
        fmodel.summary()
    return fmodel

#%%
#function that returns a custom model
def getCustomModel(input_shape,showsummary=False):
    #cast the input layer
    inp = Input(shape=input_shape)
    #block1
    cmodel = Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='linear')(inp)
    cmodel = BatchNormalization()(cmodel)
    cmodel = Activation('relu')(cmodel)
    cmodel = MaxPool2D((2,2))(cmodel)
    #block2
    cmodel = Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='linear')(cmodel)
    cmodel = BatchNormalization()(cmodel)
    cmodel = Activation('relu')(cmodel)
    cmodel = MaxPool2D((2,2))(cmodel)
    #block3
    cmodel = Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='linear')(cmodel)
    cmodel = BatchNormalization()(cmodel)
    cmodel = Activation('relu')(cmodel)
    cmodel = MaxPool2D((2,2))(cmodel)
    #block4
    cmodel = Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='linear',bias_initializer=keras.initializers.Constant(value=-1.99))(cmodel)
    cmodel = BatchNormalization()(cmodel)
    cmodel = Activation('relu')(cmodel)
    cmodel = MaxPool2D((2,2))(cmodel)
    #final chunk
    cmodel = Flatten()(cmodel)
    cmodel = Dense(64)(cmodel)
    cmodel = Dropout(0.5)(cmodel)
    cmodel = BatchNormalization()(cmodel)
    cmodel = Activation('relu')(cmodel)
    cmodel = Dense(128)(cmodel)
    cmodel = Dropout(0.33)(cmodel)
    cmodel = BatchNormalization()(cmodel)
    cmodel = Activation('relu')(cmodel)
    output_cmodel = Dense(24,activation='sigmoid',name='FashionCustom')(cmodel)
    
    customModel = Model([inp],[output_cmodel],name='FashionCustom')
    
    if showsummary:
        #model summary
        customModel.summary()
    return customModel

#%%
#train time variables
modelpath = r'FashionNet.h5'
checkpoint = ModelCheckpoint(filepath=modelpath,monitor='val_loss',verbose=1,save_best_only=True,mode='min')
reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.9,patience=3,min_lr=1e-9,mode='min',cooldown=3,verbose=1)
callbacksList = [checkpoint,reduceLR]

#%%
#the focal loss function
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        pt_1 = K.clip(pt_1, 1e-3, .999)
        pt_0 = K.clip(pt_0, 1e-3, .999)
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

#%%
#the f1 score evaluation metric
def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

#%%
#the f1 score based loss function
def f1_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

#%%
def hamming_multilabel_loss(y_true, y_pred):
    # Avoid divide by 0
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # Multi-task loss
    return K.mean(K.sum(- y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred), axis=1))

#%%
#main training cell
if os.path.exists(modelpath):
    FashionNet = load_model(modelpath,compile=False)
    print('Model loaded from disk.')
else:
    print('New model will be trained.')
    FashionNet = getCustomModel(inputshape,showsummary=False)
    #the optimizers and model compilation
    theOptimizer = keras.optimizers.Adam(lr=0.001)
    #theloss = 'binary_crossentropy'
    FashionNet.compile(loss=focal_loss(),optimizer=theOptimizer,metrics=[hamming_multilabel_loss])
    #fit generator
    loss_history = FashionNet.fit_generator(trainGen,
                                      steps_per_epoch=40,
                                      validation_data=(v_imgs,v_labels),
                                      validation_steps=9,
                                      epochs=10)

#%%
#fine tuning cell with callbacks
if FashionNet:
    #the optimizers and model compilation
    theOptimizer = keras.optimizers.Adam(lr=0.0001)
    #theloss = 'binary_crossentropy'
    FashionNet.compile(loss=focal_loss(),optimizer=theOptimizer,metrics=[hamming_multilabel_loss])
    #fit generator
    loss_history_ft = FashionNet.fit_generator(trainGen,
                                      steps_per_epoch=40,
                                      validation_data=(v_imgs,v_labels),
                                      validation_steps=9,
                                      epochs=50,
                                      callbacks=callbacksList)

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
            break

inferenceTestset(FashionNet,testData,imgfolpath=image_folpath,thresh=0.4)
