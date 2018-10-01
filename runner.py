import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")

from sklearn.model_selection import train_test_split
#from tqdm import tqdm_notebook , tnrange
from skimage.io import imread, imshow #, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.preprocessing.image import array_to_img, img_to_array, load_img,save_img
from keras.models import Model, load_model, save_model
from keras.layers import Input
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from models import build_Unet_Resnet_custom, my_iou_metric, my_iou_metric_2, lovasz_loss, iou_metric_batch

import time
t_start = time.time()

version = 4
basic_name = 'models/Unet_resnet_v3'
save_model_entropy_loss_name = basic_name+"_entropy_loss"+ '.model'
save_model_lovasz_loss_name = basic_name+"_lovasz_loss"+ '.model'
submission_file = basic_name + '.csv'

print(save_model_entropy_loss_name)
print(save_model_lovasz_loss_name)
print(submission_file)

img_size_ori = 101
img_size_target = 101
data_path = "input/"

def upsample(img):  # not used
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)


def downsample(img):  # not used
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)


# Loading of training/testing ids and depths
train_df = pd.read_csv(data_path+"train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv(data_path+"depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]

train_df["images"] = [np.array(load_img("input/train/images/{}.png".format(idx), grayscale=True)) / 255 for idx
                      in train_df.index]

train_df["masks"] = [np.array(load_img("input/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in
                     train_df.index]



#add coverage
train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)

def cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i:
            return i

train_df["coverage_class"] = train_df.coverage.map(cov_to_class)

#split_test_train
ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = train_test_split(
    train_df.index.values,
    np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1),
    np.array(train_df.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1),
    train_df.coverage.values,
    train_df.z.values,
    test_size=0.2, stratify=train_df.coverage_class, random_state= 1234)


def augment_data(images):
    flipped_lr = np.array([np.fliplr(img) for img in images])
    flipped_ud = np.array([np.flipud(img) for img in images])
    rotate_clock1 = np.array([np.rot90(img, axes=(0,1)) for img in images])
    rotate_clock2 = np.array([np.rot90(img, k=1, axes=(0,1)) for img in images])
    rotate_clock3 = np.array([np.rot90(img, k=2, axes=(0,1)) for img in images])
    return (images, flipped_lr, flipped_ud, rotate_clock1, rotate_clock2, rotate_clock3)
    #return augmented_imgs

#Data augmentation
#x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
#y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)

print(x_train.shape)
x_train = np.concatenate(augment_data(x_train))
y_train = np.concatenate(augment_data(y_train))
print(x_train.shape)
print(y_train.shape)



#Build Model 1
input_layer = Input((img_size_target, img_size_target, 1))
output_layer = build_Unet_Resnet_custom(input_layer, 16,0.5)
model1 = Model(input_layer, output_layer)
c = optimizers.adam(lr = 0.01)
model1.compile(loss="binary_crossentropy", optimizer=c, metrics=[my_iou_metric])

print(model1.summary())

model_checkpoint = ModelCheckpoint(save_model_entropy_loss_name,monitor='my_iou_metric',
                                   mode = 'max', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='my_iou_metric', mode = 'max',factor=0.5, patience=10, min_lr=0.0001, verbose=1)

epochs = 100
batch_size = 64
history = model1.fit(x_train, y_train,
                    validation_data=[x_valid, y_valid],
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[ model_checkpoint,reduce_lr],
                    verbose=2)

# Create Model 2
model1 = load_model(save_model_entropy_loss_name,custom_objects={'my_iou_metric': my_iou_metric})

# remove layter activation layer and use losvasz loss
input_x = model1.layers[0].input
output_layer = model1.layers[-1].input
model = Model(input_x, output_layer)
c = optimizers.adam(lr = 0.01)

# lovasz_loss need input range (-∞，+∞), so cancel the last "sigmoid" activation
# Then the default threshod for pixel prediction is 0 instead of 0.5, as in my_iou_metric_2.
model.compile(loss=lovasz_loss, optimizer=c, metrics=[my_iou_metric_2])

print(model.summary())

early_stopping = EarlyStopping(monitor='val_my_iou_metric_2', mode = 'max',patience=25, verbose=1)
model_checkpoint = ModelCheckpoint(save_model_lovasz_loss_name,monitor='val_my_iou_metric_2',
                                   mode = 'max', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric_2', mode = 'max',factor=0.5, patience=10, min_lr=0.0001, verbose=1)
epochs = 100
batch_size = 64

history = model.fit(x_train, y_train,
                    validation_data=[x_valid, y_valid],
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[ model_checkpoint,reduce_lr,early_stopping],
                    verbose=2)

model = load_model(save_model_lovasz_loss_name,custom_objects={'my_iou_metric_2': my_iou_metric_2,
                                                   'lovasz_loss': lovasz_loss})


def predict_result(model,x_test,img_size_target): # predict both orginal and reflect x
    #x_test_augmented = augment_data(x_test)
    #preds = np.zeros_like(x_test)
    #print (len(x_test_augmented))
    #for i in range(len(x_test_augmented)):
    # TODO use all augmentations
    preds_test = model.predict(x_test).reshape(-1, img_size_target, img_size_target)
    x_test_reflect =  np.array([np.fliplr(x) for x in x_test])
    preds_test2_refect = model.predict(x_test_reflect).reshape(-1, img_size_target, img_size_target)
    preds_test += np.array([ np.fliplr(x) for x in preds_test2_refect] )
    return preds_test/2

preds_valid = predict_result(model,x_valid,img_size_target)


## Scoring for last model, choose threshold by validation data
thresholds_ori = np.linspace(0.3, 0.7, 31)
# Reverse sigmoid function: Use code below because the  sigmoid activation was removed
thresholds = np.log(thresholds_ori/(1-thresholds_ori))

# ious = np.array([get_iou_vector(y_valid, preds_valid > threshold) for threshold in tqdm_notebook(thresholds)])
# print(ious)
ious = np.array([iou_metric_batch(y_valid, preds_valid > threshold) for threshold in thresholds])
print(ious)


# instead of using default 0 as threshold, use validation data to find the best threshold.
threshold_best_index = np.argmax(ious)
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]

# plt.plot(thresholds, ious)
# plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
# plt.xlabel("Threshold")
# plt.ylabel("IoU")
# plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
# plt.legend()

def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

x_test = np.array([(np.array(load_img("input/test/images/{}.png".format(idx), grayscale = True))) / 255 for idx in test_df.index]).reshape(-1, img_size_target, img_size_target, 1)
preds_test = predict_result(model,x_test,img_size_target)
t1 = time.time()
pred_dict = {idx: rle_encode(np.round(downsample(preds_test[i]) > threshold_best)) for i, idx in enumerate(test_df.index.values)}
t2 = time.time()

print("Usedtime = {t2-t1} s")

sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv("input/submission_file")

t_finish = time.time()
print("Kernel run time = %f hours"%((t_finish - t_start)/3600))
