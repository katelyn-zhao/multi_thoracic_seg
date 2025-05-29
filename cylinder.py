import random
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import nibabel
import cv2
import numpy as np
import nibabel as nib
from nibabel import load
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import rotate
from skimage.exposure import rescale_intensity
# from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.utils import normalize
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.model_selection import KFold
from tensorflow.keras.models import Model
from collections import Counter
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss
from tensorflow.keras.callbacks import ReduceLROnPlateau


# Defining Metrics
##############################################################################################

num_classes = 7

# Dice Coefficient
def dice_coef(y_true, y_pred):
    total_dice = 0.0
    num_class = 0.0
    for class_idx in range(num_classes):
        y_true_class = y_true[..., class_idx]
        y_pred_class = y_pred[..., class_idx]
        y_true_f = tf.keras.backend.flatten(y_true_class)
        y_pred_f = tf.keras.backend.flatten(y_pred_class)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        dice = (2. * intersection) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1e-7)
        total_dice = total_dice + dice
        num_class = num_class + 1.0
    mean_dice_score = total_dice / num_class
    return mean_dice_score

# True Positive Rate (TPR)
def tpr(y_true, y_pred, threshold=0.5):
    total_tpr = 0
    num_class = 0
    for class_idx in range(num_classes):
        y_true_class = y_true[..., class_idx]
        y_pred_class = y_pred[..., class_idx]
        y_pred_pos = tf.cast(y_pred_class > threshold, tf.float32)
        y_true_pos = tf.cast(y_true_class > threshold, tf.float32)
        true_pos = tf.reduce_sum(tf.cast(tf.logical_and(y_true_pos == 1, y_pred_pos == 1), tf.float32))
        actual_pos = tf.reduce_sum(tf.cast(y_true_pos, tf.float32))
        tpr = true_pos / (actual_pos + tf.keras.backend.epsilon())
        total_tpr += tpr
        num_class += 1
    mean_tpr = total_tpr / num_class
    return mean_tpr

# False Positive Rate (FPR)
def fpr(y_true, y_pred, threshold=0.5):
    total_fpr = 0
    num_class = 0
    for class_idx in range(num_classes):
        y_true_class = y_true[..., class_idx]
        y_pred_class = y_pred[..., class_idx]
        y_pred_pos = tf.cast(y_pred_class > threshold, tf.float32)
        y_true_neg = tf.cast(y_true_class <= threshold, tf.float32)
        false_pos = tf.reduce_sum(tf.cast(tf.logical_and(y_true_neg == 1, y_pred_pos == 1), tf.float32))
        actual_neg = tf.reduce_sum(tf.cast(y_true_neg, tf.float32))
        fpr = false_pos / (actual_neg + tf.keras.backend.epsilon())
        total_fpr += fpr
        num_class += 1
    mean_fpr = total_fpr / num_class
    return mean_fpr

# Metrics for evaluation/prediction
def dice_coef_p(y_true, y_pred, smooth=1.):
    total_dice_score = 0
    num_class = 0
    for class_idx in range(num_classes):
        intersection = np.sum(y_true[..., class_idx] * y_pred[..., class_idx])
        union = np.sum(y_true[..., class_idx]) + np.sum(y_pred[..., class_idx])
        dice_score = (2.0 * intersection + smooth) / (union + smooth)
        total_dice_score += dice_score
        num_class += 1
    mean_dice_score = total_dice_score / num_class
    return mean_dice_score

def tpr_p(y_true, y_pred, threshold=0.5):
    total_tpr = 0
    num_class = 0
    for class_idx in range(num_classes):
        y_pred_thresh = (y_pred[..., class_idx] >= threshold)
        tp = np.sum((y_pred_thresh == 1) & (y_true[..., class_idx] == 1))
        fn = np.sum((y_pred_thresh == 0) & (y_true[..., class_idx] == 1))
        if (tp == 0):
            tpr = 0
        else:
            tpr = tp / (tp + fn)
        total_tpr += tpr
        num_class += 1
    mean_tpr = total_tpr / num_class
    return mean_tpr


def fpr_p(y_true, y_pred, threshold=0.5):
    total_fpr = 0
    num_class = 0
    for class_idx in range(num_classes):
        y_pred_thresh = (y_pred[..., class_idx] >= threshold)
        fp = np.sum((y_pred_thresh == 1) & (y_true[..., class_idx] == 0))
        tn = np.sum((y_pred_thresh == 0) & (y_true[..., class_idx] == 0))
        if (fp == 0):
            fpr = 0
        else:
            fpr = fp / (fp + tn)
        total_fpr += fpr
        num_class += 1
    mean_fpr = total_fpr / num_class
    return mean_fpr

# Combined Loss Function
def combined_loss(y_true, y_pred):
    cce_loss = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred)
    dice_loss = 1 - dice_coef(y_true, y_pred)
    return cce_loss + dice_loss

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1, n_classes])
    y_pred_f = tf.reshape(y_pred, [-1, n_classes])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0) + smooth)
    return 1.0 - tf.reduce_mean(score)


# Model Definition
##############################################################################################

def multi_unet_model(n_classes=7, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):
    # Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs

    # Contraction path
    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    c4 = BatchNormalization()(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5) 
    c5 = BatchNormalization()(c5)
    p5 = MaxPooling2D(pool_size=(2, 2))(c5)
    
    c6 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p5)
    c6 = BatchNormalization()(c6)
    c6 = Dropout(0.3)(c6)
    c6 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    c6 = BatchNormalization()(c6)
    
    # Expansive path 
    u7 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c5])
    c7 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    c7 = BatchNormalization()(c7)
     
    u8 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c4])
    c8 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = BatchNormalization()(c8)
    c8 = Dropout(0.2)(c8)
    c8 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    c8 = BatchNormalization()(c8)
     
    u9 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c3])
    c9 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = BatchNormalization()(c9)
    c9 = Dropout(0.2)(c9)
    c9 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    c9 = BatchNormalization()(c9)
     
    u10 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c9)
    u10 = concatenate([u10, c2])
    c10 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u10)
    c10 = BatchNormalization()(c10)
    c10 = Dropout(0.1)(c10)
    c10 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c10)
    c10 = BatchNormalization()(c10)
     
    u11 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c10)
    u11 = concatenate([u11, c1], axis=3)
    c11 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u11)
    c11 = BatchNormalization()(c11)
    c11 = Dropout(0.1)(c11)
    c11 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c11)
    c11 = BatchNormalization()(c11)
     
    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c11)     
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss=combined_loss, metrics=[dice_coef, tpr, fpr])
    model.summary()
    
    return model

# Image Processing
##############################################################################################

#Number of classes for segmentation
n_classes= 7

#Capture training image info as a list
image_directory = 'C:/Users/Mittal/Desktop/thoracic_seg/raw_images/'
mask_directory = 'C:/Users/Mittal/Desktop/thoracic_seg/segmentations/'
# predictions_directory = 'C:/Users/Mittal/Desktop/thoracic_seg/unet_niipredictions/'

image_dataset = []
mask_dataset = []
# prediction_dataset = []
sliced_image_dataset = []
sliced_mask_dataset = []
# sliced_prediction_dataset = []
image_names = []
sliced_image_names = []

images = sorted(os.listdir(image_directory))
for i, image_name in enumerate(images):    
    if (image_name.split('.')[1] == 'nii'):
        image = nib.load(image_directory+image_name)
        image = np.array(image.get_fdata())
        image_dataset.append(np.array(image))
        image_names.append(image_name.split('.')[0])

masks = sorted(os.listdir(mask_directory))
for i, image_name in enumerate(masks):
    if (image_name.split('.')[1] == 'nii'):
        image = nib.load(mask_directory+image_name)
        image = np.array(image.get_fdata())
        mask_dataset.append(np.array(image))

# predictions = sorted(os.listdir(predictions_directory))
# for i, image_name in enumerate(predictions):
#     if (image_name.split('.')[1] == 'nii'):
#         image = nib.load(predictions_directory+image_name)
#         image = np.array(image.get_fdata())
#         prediction_dataset.append(np.array(image))

original_data_size = 0

for i in range(len(image_dataset)):
    if image_dataset[i].shape[2] < 30:
        for j in range(6, mask_dataset[i].shape[2]-6):
            # thoracic_body = np.where(prediction_dataset[i][:,:,j] == 1, 1, 0)
            # thoracic_body = 1 - thoracic_body

            # new_image = image_dataset[i][:,:,j]*thoracic_body

            # new_mask = np.copy(mask_dataset[i][:,:,j])
            # new_mask[mask_dataset[i][:,:,j] == 1] = 0

            sliced_image_dataset.append(image_dataset[i][:,:,j])
            sliced_mask_dataset.append(mask_dataset[i][:,:,j])
            sliced_image_names.append(image_names[i] + '-' + str(j))
            original_data_size += 1
            # #rotation
            # cw = random.randint(0,1)
            # angle = random.randint(5,10)
            # #contrast adjustment
            # adjust = random.randint(0,1)
            # contrast = random.randint(1,2)
            # #reflection
            # reflect = random.randint(0,1)
            # #applying changes
            # if adjust and cw == 1:
            #     augmented_image = rotate(cv2.convertScaleAbs(new_image, alpha = contrast, beta = 0), angle, reshape = False, order=1)
            #     augmented_mask = rotate(new_mask, angle, reshape = False, order=0)
                
            #     sliced_image_dataset.append(augmented_image)
            #     sliced_mask_dataset.append(augmented_mask)
            #     sliced_image_names.append(image_names[i] + '-' + str(j) + '-aug')
            # if adjust and cw == 0:
            #     augmented_image = rotate(cv2.convertScaleAbs(new_image, alpha = contrast, beta = 0), angle * -1, reshape = False, order=1)
            #     augmented_mask = rotate(new_mask, angle * -1, reshape = False, order=0)
                
            #     sliced_image_dataset.append(augmented_image)
            #     sliced_mask_dataset.append(augmented_mask)
            #     sliced_image_names.append(image_names[i] + '-' + str(j) + '-aug')
    else:
        for j in range(10, mask_dataset[i].shape[2]-10):
            # thoracic_body = np.where(mask_dataset[i][:,:,j] == 1, 1, 0)
            # thoracic_body = 1 - thoracic_body

            # new_image = image_dataset[i][:,:,j]*thoracic_body

            # new_mask = np.copy(mask_dataset[i][:,:,j])
            # new_mask[mask_dataset[i][:,:,j] == 1] = 0

            sliced_image_dataset.append(image_dataset[i][:,:,j])
            sliced_mask_dataset.append(mask_dataset[i][:,:,j])
            sliced_image_names.append(image_names[i] + '-' + str(j))
            original_data_size += 1
            # #rotation
            # cw = random.randint(0,1)
            # angle = random.randint(5,10)
            # #contrast adjustment
            # adjust = random.randint(0,1)
            # contrast = random.randint(1,2)
            # #applying changes
            # if adjust and cw == 1:
            #     augmented_image = rotate(cv2.convertScaleAbs(new_image, alpha = contrast, beta = 0), angle, reshape = False, order=1)
            #     augmented_mask = rotate(new_mask, angle, reshape = False, order=0)
                
            #     sliced_image_dataset.append(augmented_image)
            #     sliced_mask_dataset.append(augmented_mask)
            #     sliced_image_names.append(image_names[i] + '-' + str(j) + '-aug')
            # if adjust and cw == 0:
            #     augmented_image = rotate(cv2.convertScaleAbs(new_image, alpha = contrast, beta = 0), angle * -1, reshape = False, order=1)
            #     augmented_mask = rotate(new_mask, angle * -1, reshape = False, order=0)
                
            #     sliced_image_dataset.append(augmented_image)
            #     sliced_mask_dataset.append(augmented_mask)
            #     sliced_image_names.append(image_names[i] + '-' + str(j) + '-aug')


sliced_image_dataset = np.array(sliced_image_dataset)
sliced_mask_dataset = np.array(sliced_mask_dataset)
image_names = np.array(image_names)
sliced_image_names = np.array(sliced_image_names)

print(f'Original Data Size: {original_data_size}')
print(f'Dataset Size: {len(sliced_image_dataset)}')

#Sanity check, view a few images
# image_number = random.randint(0, len(sliced_image_dataset))
# plt.figure(figsize=(12, 6))
# plt.subplot(121)
# plt.imshow(sliced_image_dataset[image_number], cmap='gray')
# plt.subplot(122)
# plt.imshow(sliced_mask_dataset[image_number], cmap='gray')
# plt.show()

#Encode labels... but multi dim array so need to flatten, encode and reshape
labelencoder = LabelEncoder()
n, h, w = sliced_mask_dataset.shape
sliced_masks_reshaped = sliced_mask_dataset.reshape(-1,1)
sliced_masks_reshaped_encoded = labelencoder.fit_transform(sliced_masks_reshaped)
sliced_masks_encoded_original_shape = sliced_masks_reshaped_encoded.reshape(n, h, w)

print(f'Labels: {np.unique(sliced_masks_encoded_original_shape)}')

sliced_image_dataset = np.expand_dims(sliced_image_dataset, axis=3)
sliced_image_dataset = normalize(sliced_image_dataset, axis=1)

sliced_mask_dataset = np.expand_dims(sliced_masks_encoded_original_shape, axis=3)

f = open(f"C:/Users/Mittal/Desktop/thoracic_seg/outputs/multi_unet_output.txt", "a")
print("original image dataset: ", original_data_size, file=f)
print("sliced image dataset: ", len(sliced_image_dataset), file=f)
f.close()

def manual_class_weight(labels):
    class_count = Counter(labels)
    total = sum(class_count.values())
    classes = sorted(class_count.keys())
    class_weights = [total / (len(class_count) * class_count[cls]) for cls in classes]
    return class_weights

class_weights = manual_class_weight(sliced_masks_reshaped_encoded)
class_weights /= np.sum(class_weights)

f = open(f"C:/Users/Mittal/Desktop/thoracic_seg/outputs/multi_unet_output.txt", "a")
print("Class weights:", class_weights, file=f)
f.close()


# Training and Prediction
##############################################################################################

def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

n_splits = 5

kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

# Iterate over each fold
for i, (train_index, test_index) in enumerate(kf.split(sliced_image_dataset, sliced_mask_dataset)):
    X_train, X_test = sliced_image_dataset[train_index], sliced_image_dataset[test_index]
    y_train, y_test = sliced_mask_dataset[train_index], sliced_mask_dataset[test_index]
    name_test = np.array(sliced_image_names)[test_index]
    y_train_cat = to_categorical(y_train, num_classes=n_classes)
    y_test_cat = to_categorical(y_test, num_classes=n_classes)

    IMG_HEIGHT = 256
    IMG_WIDTH  = 256
    IMG_CHANNELS = 1

    model = get_model()

    checkpoint = ModelCheckpoint(f'C:/Users/Mittal/Desktop/thoracic_seg/models/multi_unet_model2_{i}.h5', monitor='val_loss', save_best_only=True)
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                 factor=0.5, 
                                 patience=10, 
                                 verbose=1, 
                                 min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=1)


    history = model.fit(X_train, y_train_cat, 
                        batch_size=64, 
                        verbose=1, 
                        epochs=1000, 
                        validation_data=(X_test, y_test_cat), 
                        shuffle=False,
                        callbacks=[checkpoint, lr_reduction, early_stopping])
                        
    #Evaluate the model
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], color='r')
    plt.plot(history.history['val_loss'])
    plt.ylabel('Losses')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val.'], loc='upper right')
    plt.subplot(1,2,2)
    plt.plot(history.history['dice_coef'], color='r')
    plt.plot(history.history['val_dice_coef'])
    plt.ylabel('dice_coef')
    plt.xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(f'C:/Users/Mittal/Desktop/thoracic_seg/outputs/multi_unet_process2_{i}.png')
    plt.close()

    max_dice_coef = max(history.history['dice_coef'])
    max_val_dice_coef = max(history.history['val_dice_coef'])
    max_tpr = max(history.history['tpr'])
    min_fpr = min(history.history['fpr'])

    f = open(f'C:/Users/Mittal/Desktop/thoracic_seg/outputs/multi_unet_output.txt', "a")
    print("FOLD------------------------------------------", file=f)
    print("Max Dice Score: ", max_dice_coef, file=f)
    print("Max Val Dice Score: ", max_val_dice_coef, file=f)
    print("Max TPR: ", max_tpr, file=f)
    print("Max FPR: ", min_fpr, file=f)
    f.close()
        
    model.load_weights(f'C:/Users/Mittal/Desktop/thoracic_seg/models/multi_unet_model2_{i}.h5')

    dice_scores = []
    tprs = []
    fprs = []

    for z in range(50):
        test_img_number = random.randint(0, len(X_test)-1)
        test_img = X_test[test_img_number]
        ground_truth = y_test[test_img_number]
        ground_truth_cat = y_test_cat[test_img_number]
        test_img_norm = test_img[:,:,0][:,:,None]
        test_img_input = np.expand_dims(test_img_norm, 0)
        prediction = (model.predict(test_img_input))
        predicted_img = np.argmax(prediction, axis=3)[0,:,:]

        dice_score = dice_coef_p(ground_truth_cat, prediction)
        pred_tpr = tpr_p(ground_truth_cat, prediction)
        pred_fpr = fpr_p(ground_truth_cat, prediction)
        dice_scores.append(dice_score)
        tprs.append(pred_tpr)
        fprs.append(pred_fpr)

        plt.figure(figsize=(16, 8))
        plt.subplot(131)
        plt.title('Testing Image')
        plt.imshow(test_img[:,:,0], cmap='gray')
        plt.subplot(132)
        plt.title('Testing Label')
        plt.imshow(ground_truth[:,:,0], cmap='jet')
        plt.subplot(133)
        plt.title('Prediction on test image')
        plt.imshow(predicted_img, cmap='jet')
        plt.savefig(f'C:/Users/Mittal/Desktop/thoracic_seg/multi_predictions2/fold{i}_{name_test[test_img_number]}.png')
        plt.close()


    # filewise_predictions = {filename: [] for filename in image_names}
    # for idx, filename in enumerate(image_names):
    #     num_slices_per_image = image_dataset[idx].shape[2]  # Assuming image_dataset is accessible here
    #     start_index = sum(image_dataset[i].shape[2] for i in range(idx))  # Start index for slices of this image
    #     for z in range(num_slices_per_image):
    #         test_img = sliced_image_dataset[start_index + z]
    #         ground_truth = sliced_mask_dataset[start_index + z]
    #         ground_truth_cat = to_categorical(sliced_mask_dataset, 7)[start_index + z]
    #         test_img_norm = test_img[:,:,0][:,:,None]
    #         test_img_input = np.expand_dims(test_img_norm, 0)
    #         prediction = (model.predict(test_img_input))
    #         predicted_img = np.argmax(prediction, axis=3)[0,:,:]

    #         dice_score = dice_coef_p(ground_truth_cat, prediction)
    #         pred_tpr = tpr_p(ground_truth_cat, prediction)
    #         pred_fpr = fpr_p(ground_truth_cat, prediction)
    #         dice_scores.append(dice_score)
    #         tprs.append(pred_tpr)
    #         fprs.append(pred_fpr)

    #         filewise_predictions[filename].append(prediction)
    #         if z > 15 and z < 19:
    #             plt.figure(figsize=(16, 8))
    #             plt.subplot(131)
    #             plt.title('Testing Image')
    #             plt.imshow(test_img[:,:,0], cmap='gray')
    #             plt.subplot(132)
    #             plt.title('Testing Label')
    #             plt.imshow(ground_truth[:,:,0], cmap='jet')
    #             plt.subplot(133)
    #             plt.title('Prediction on test image')
    #             plt.imshow(predicted_img, cmap='jet')
    #             plt.savefig(f'C:/Users/Mittal/Desktop/thoracic_seg/multi_predictions/fold{i}_{filename}_{z}.png')
    #             plt.close()

    # for filename, predictions in filewise_predictions.items():
    #     if len(predictions) > 0:  # Check if predictions are available
    #         three_d_predictions_volume = np.stack(predictions, axis=-1)
    #         affine = np.eye(4)
    #         nii_img = nib.Nifti1Image(three_d_predictions_volume, affine)
    #         nib.save(nii_img, f'C:/Users/Mittal/Desktop/thoracic_seg/multi_niipredictions/{filename}.nii')
    #     else:
    #         print(f"No predictions available for {filename}.")

    f = open(f'C:/Users/Mittal/Desktop/thoracic_seg/outputs/multi_unet_output.txt', "a")
    print("Average Prediction Dice Score: ", np.mean(dice_scores), file=f)
    print("Average Prediction TPR: ", np.mean(tprs), file=f)
    print("Average Prediction FPR: ", np.mean(fprs), file=f)
    f.close()