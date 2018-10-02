import numpy as np
import pandas as pd
import time
from random import randint

import matplotlib.pyplot as plt
import gc
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")

from sklearn.model_selection import train_test_split
from skimage.transform import resize
from keras.preprocessing.image import load_img
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
import model2


img_size_ori = 101
img_size_target = 128


def upsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)
    # res = np.zeros((img_size_target, img_size_target), dtype=img.dtype)
    # res[:img_size_ori, :img_size_ori] = img
    # return res


def downsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)
    # return img[:img_size_ori, :img_size_ori]


train_df = pd.read_csv("input/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("input/depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]


train_df["images"] = [np.array(load_img("input/train/images/{}.png".format(idx), grayscale=True)) / 255 for idx in train_df.index]
train_df["masks"] = [np.array(load_img("input/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in train_df.index]

train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)


def cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i:
            return i

train_df["coverage_class"] = train_df.coverage.map(cov_to_class)

ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = train_test_split(
    train_df.index.values,
    np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1),
    np.array(train_df.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1),
    train_df.coverage.values,
    train_df.z.values,
    test_size=0.2, stratify=train_df.coverage_class, random_state=1337)

K.clear_session()
model = model2.get_unet_resnet(input_shape=(img_size_target,img_size_target,3))
model.compile(loss=model2.bce_dice_loss, optimizer='adam', metrics=[model2.my_iou_metric])

x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)

x_train = np.repeat(x_train,3,axis=3)
x_valid = np.repeat(x_valid,3,axis=3)

model_checkpoint = ModelCheckpoint("models/RESNet50_pretrained.model",monitor='val_my_iou_metric',
                                   mode = 'max', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(factor=0.1, patience=4, min_lr=0.00001, verbose=1)

epochs = 1
batch_size = 32

history = model.fit(x_train, y_train,
                    validation_data=[x_valid, y_valid],
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[model_checkpoint, reduce_lr],shuffle=True,verbose=2)


del ids_train, x_train, y_train, cov_train,depth_train,depths_df,depth_test
gc.collect()

model.load_weights('models/RESNet50_pretrained.model')


def predict_result(model,x_test,img_size_target): # predict both orginal and reflect x
    x_test_reflect =  np.array([np.fliplr(x) for x in x_test])
    preds_test1 = model.predict([x_test]).reshape(-1, img_size_target, img_size_target)
    preds_test2_refect = model.predict([x_test_reflect]).reshape(-1, img_size_target, img_size_target)
    preds_test2 = np.array([ np.fliplr(x) for x in preds_test2_refect] )
    preds_avg = (preds_test1 +preds_test2)/2
    return preds_avg


preds_valid = predict_result(model,x_valid,img_size_target)
preds_valid = np.array([downsample(x) for x in preds_valid])
y_valid_ori = np.array([downsample(x) for x in y_valid])


thresholds = np.linspace(0.3, 0.7, 31)
ious = np.array([model2.iou_metric_batch(y_valid_ori, np.int32(preds_valid > threshold)) for threshold in thresholds])

threshold_best_index = np.argmax(ious)
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]

del x_valid, y_valid,preds_valid,y_valid_ori,train_df
gc.collect()

def rle_encode(im):
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

batch_size = 500
preds_test = []
i = 0
while i < test_df.shape[0]:
    print('Images Processed:',i)
    index_val = test_df.index[i:i+batch_size]
    depth_val = test_df.z[i:i+batch_size]
    x_test = np.array([upsample(np.array(load_img("input/test/images/{}.png".format(idx), grayscale=True))) / 255 for idx in (index_val)]).reshape(-1, img_size_target, img_size_target, 1)
    x_test = np.repeat(x_test,3,axis=3)
    preds_test_temp = predict_result(model,x_test,img_size_target)
    if i==0:
        preds_test = preds_test_temp
    else:
        preds_test = np.concatenate([preds_test,preds_test_temp],axis=0)
#     print(preds_test.shape)
    i += batch_size


del x_test,preds_test_temp
gc.collect()



t1 = time.time()
pred_dict = {idx: rle_encode(np.round(downsample(preds_test[i]) > threshold_best)) for i, idx in enumerate(test_df.index.values)}
t2 = time.time()

print("Usedtime = {0} s".format((t2-t1)))

sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('resnet50_pretrained_submission.csv')
