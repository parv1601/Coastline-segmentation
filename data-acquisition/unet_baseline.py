import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Concatenate, add, BatchNormalization, Activation
import os
import glob
import tifffile as tiff
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# from sklearn.metrics import precision_score, recall_score, f1_score
from PIL import Image
import rasterio
from sklearn.model_selection import train_test_split;
import re
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

tf.keras.backend.clear_session()

tf.config.list_physical_devices('GPU')

def conv_block(x, filters, batchnorm=True):
    conv1 = Conv2D(filters, (3, 3), kernel_initializer='he_normal', padding='same')(x)
    if batchnorm is True:
        conv1 = BatchNormalization(axis=3)(conv1)
    conv1 = Activation('relu')(conv1)    
    conv2 = Conv2D(filters, (3, 3), kernel_initializer='he_normal', padding='same')(conv1)
    if batchnorm is True:
        conv2 = BatchNormalization(axis=3)(conv2)
    conv2 = Activation("relu")(conv2)

    return conv2

def residual_conv_block(x, filters, batchnorm=True):
    conv1 = Conv2D(filters, (3, 3), kernel_initializer='he_normal', padding='same')(x)
    if batchnorm is True:
        conv1 = BatchNormalization(axis=3)(conv1)
    conv1 = Activation('relu')(conv1)    
    conv2 = Conv2D(filters, (3, 3), kernel_initializer='he_normal', padding='same')(conv1)
    if batchnorm is True:
        conv2 = BatchNormalization(axis=3)(conv2)
    conv2 = Activation("relu")(conv2)
        
    #skip connection    
    shortcut = Conv2D(filters, kernel_size=(1, 1), kernel_initializer='he_normal', padding='same')(x)
    if batchnorm is True:
        shortcut = BatchNormalization(axis=3)(shortcut)
    shortcut = Activation("relu")(shortcut)
    respath = add([shortcut, conv2])       
    return respath

def dense_block(inputs, num_filters):
    conv1 = conv_block(inputs, num_filters)
    concat = Concatenate()([inputs, conv1])
    return concat

def residual_unet(input_shape, base_filters=64):
    inputs = Input(input_shape)
    
    # Encoder
    conv1 = residual_conv_block(inputs, base_filters)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = residual_conv_block(pool1, base_filters * 2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = residual_conv_block(pool2, base_filters * 4)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = residual_conv_block(pool3, base_filters * 8)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # Bottleneck
    conv5 = Conv2D(base_filters * 16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool4)
    conv5 = Conv2D(base_filters * 16, (3, 3), kernel_initializer='he_normal', padding='same')(conv5)
    conv5 = Activation('relu')(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    # Decoder
    up6 = Conv2DTranspose(base_filters * 8, (2, 2), strides=(2, 2), padding='same')(drop5)
    up6 = Concatenate()([up6, conv4])
    conv6 = residual_conv_block(up6, base_filters * 8)
    up7 = Conv2DTranspose(base_filters * 4, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = Concatenate()([up7, conv3])
    conv7 = residual_conv_block(up7, base_filters * 4)
    up8 = Conv2DTranspose(base_filters * 2, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = Concatenate()([up8, conv2])
    conv8 = residual_conv_block(up8, base_filters * 2)
    up9 = Conv2DTranspose(base_filters, (2, 2), strides=(2, 2), padding='same')(conv8)
    up9 = Concatenate()([up9, conv1])
    conv9 = residual_conv_block(up9, base_filters)
    
    # Output
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def dense_unet(input_shape, base_filters=64):
    inputs = Input(input_shape)

    # Encoder (dense blocks with dropout)
    conv1 = dense_block(inputs, base_filters)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = dense_block(pool1, base_filters * 2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = dense_block(pool2, base_filters * 4)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = dense_block(pool3, base_filters * 8)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottleneck (two convs + dropout)
    conv5 = Conv2D(base_filters * 16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool4)
    conv5 = Conv2D(base_filters * 16, (3, 3), kernel_initializer='he_normal', padding='same')(conv5)
    conv5 = Activation('relu')(conv5)
    drop5 = Dropout(0.5)(conv5)  # core stochastic layer

    # Decoder (transpose convs + residual blocks + dropout)
    up6 = Conv2DTranspose(base_filters * 8, (2, 2), strides=(2, 2), padding='same')(drop5)
    up6 = Concatenate()([up6, conv4])
    conv6 = dense_block(up6, base_filters * 8)

    up7 = Conv2DTranspose(base_filters * 4, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = Concatenate()([up7, conv3])
    conv7 = dense_block(up7, base_filters * 4)

    up8 = Conv2DTranspose(base_filters * 2, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = Concatenate()([up8, conv2])
    conv8 = dense_block(up8, base_filters * 2)

    up9 = Conv2DTranspose(base_filters, (2, 2), strides=(2, 2), padding='same')(conv8)
    up9 = Concatenate()([up9, conv1])
    conv9 = dense_block(up9, base_filters)

    # Output - binary segmentation (sigmoid)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def dense_unet_plus_plus(input_shape, base_filters=64):
    inputs = Input(input_shape)

    # ======================
    # Encoder (X_{i,0})
    # ======================
    x0_0 = dense_block(inputs, base_filters)
    p0 = MaxPooling2D((2, 2))(x0_0)

    x1_0 = dense_block(p0, base_filters * 2)
    p1 = MaxPooling2D((2, 2))(x1_0)

    x2_0 = dense_block(p1, base_filters * 4)
    p2 = MaxPooling2D((2, 2))(x2_0)

    x3_0 = dense_block(p2, base_filters * 8)
    p3 = MaxPooling2D((2, 2))(x3_0)

    # ======================
    # Bottleneck
    # ======================
    x4_0 = Conv2D(base_filters * 16, (3, 3), padding='same',
                  kernel_initializer='he_normal', activation='relu')(p3)
    x4_0 = Conv2D(base_filters * 16, (3, 3), padding='same',
                  kernel_initializer='he_normal', activation='relu')(x4_0)
    x4_0 = Dropout(0.5)(x4_0)

    # ======================
    # Decoder: Level 1
    # ======================
    x3_1 = dense_block(
        Concatenate()([
            x3_0,
            Conv2DTranspose(base_filters * 8, (2, 2), strides=(2, 2), padding='same')(x4_0)
        ]),
        base_filters * 8
    )

    x2_1 = dense_block(
        Concatenate()([
            x2_0,
            Conv2DTranspose(base_filters * 4, (2, 2), strides=(2, 2), padding='same')(x3_0)
        ]),
        base_filters * 4
    )

    x1_1 = dense_block(
        Concatenate()([
            x1_0,
            Conv2DTranspose(base_filters * 2, (2, 2), strides=(2, 2), padding='same')(x2_0)
        ]),
        base_filters * 2
    )

    x0_1 = dense_block(
        Concatenate()([
            x0_0,
            Conv2DTranspose(base_filters, (2, 2), strides=(2, 2), padding='same')(x1_0)
        ]),
        base_filters
    )

    # ======================
    # Decoder: Level 2
    # ======================
    x2_2 = dense_block(
        Concatenate()([
            x2_0, x2_1,
            Conv2DTranspose(base_filters * 4, (2, 2), strides=(2, 2), padding='same')(x3_1)
        ]),
        base_filters * 4
    )

    x1_2 = dense_block(
        Concatenate()([
            x1_0, x1_1,
            Conv2DTranspose(base_filters * 2, (2, 2), strides=(2, 2), padding='same')(x2_1)
        ]),
        base_filters * 2
    )

    x0_2 = dense_block(
        Concatenate()([
            x0_0, x0_1,
            Conv2DTranspose(base_filters, (2, 2), strides=(2, 2), padding='same')(x1_1)
        ]),
        base_filters
    )

    # ======================
    # Decoder: Level 3
    # ======================
    x1_3 = dense_block(
        Concatenate()([
            x1_0, x1_1, x1_2,
            Conv2DTranspose(base_filters * 2, (2, 2), strides=(2, 2), padding='same')(x2_2)
        ]),
        base_filters * 2
    )

    x0_3 = dense_block(
        Concatenate()([
            x0_0, x0_1, x0_2,
            Conv2DTranspose(base_filters, (2, 2), strides=(2, 2), padding='same')(x1_2)
        ]),
        base_filters
    )

    # ======================
    # Decoder: Level 4
    # ======================
    x0_4 = dense_block(
        Concatenate()([
            x0_0, x0_1, x0_2, x0_3,
            Conv2DTranspose(base_filters, (2, 2), strides=(2, 2), padding='same')(x1_3)
        ]),
        base_filters
    )
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(x0_4)

    model = Model(inputs, outputs)
    return model


def load_data_new(image_dir, mask_dir):
    import os
    import numpy as np
    import rasterio

    # Check directories
    if not os.path.exists(image_dir):
        print(f"Image directory {image_dir} does not exist.")
        return None, None

    if not os.path.exists(mask_dir):
        print(f"Mask directory {mask_dir} does not exist.")
        return None, None

    # Get all image files
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif')])

    if len(image_files) == 0:
        print(f"No .tif images found in {image_dir}")
        return None, None

    images = []
    masks = []

    for file in image_files:
        img_path = os.path.join(image_dir, file)
        mask_path = os.path.join(mask_dir, file)

        # Ensure matching mask exists
        if not os.path.exists(mask_path):
            print(f"Mask missing for {file}")
            continue

        # ---- READ IMAGE (MULTI-BAND) ----
        with rasterio.open(img_path) as src:
            img = src.read()   # (C, H, W)
            img = np.transpose(img, (1, 2, 0))  # → (H, W, C)

        # ---- READ MASK ----
        with rasterio.open(mask_path) as src:
            mask = src.read(1)  # (H, W)

        # ---- NORMALIZE IMAGE (PER CHANNEL) ----
        img = img.astype(np.float32)

        for c in range(img.shape[-1]):
            band = img[..., c]
            min_val = band.min()
            max_val = band.max()
            if max_val - min_val > 0:
                img[..., c] = (band - min_val) / (max_val - min_val)
            else:
                img[..., c] = 0.0

        # ---- BINARIZE MASK ----
        mask = (mask > 0).astype(np.float32)

        # ---- ADD CHANNEL DIM TO MASK ----
        mask = np.expand_dims(mask, axis=-1)  # (H, W, 1)

        images.append(img)
        masks.append(mask)

    if len(images) == 0:
        print("No valid image-mask pairs found.")
        return None, None

    images = np.array(images)
    masks = np.array(masks)

    print("Loaded data:")
    print("Images shape:", images.shape)
    print("Masks shape:", masks.shape)

    return images, masks


import os
import numpy as np
import rasterio
import tifffile as tiff
from sklearn.metrics import precision_score, recall_score, f1_score

def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def iou(y_true, y_pred, smooth=1):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def save_image(image, filepath):
    # Remove channel if single channel
    if len(image.shape) == 3 and image.shape[-1] == 1:
        image = image.squeeze(-1)

    # Clip to [0,1]
    image = np.clip(image, 0, 1)

    image = (image * 255).astype(np.uint8)

    tiff.imwrite(filepath, image)

def predict1(model, mask_dir, image_dir, output_dir, number=-1):

    dice_scores = []
    iou_scores = []
    precisions = []
    recalls = []
    f1_scores = []

    os.makedirs(output_dir, exist_ok=True)

    # Use same filenames as images
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif')])

    count = 0

    for file in image_files:
        image_path = os.path.join(image_dir, file)
        mask_path = os.path.join(mask_dir, file)

        if not os.path.exists(mask_path):
            print(f"Mask missing for {file}")
            continue

        # ---- READ IMAGE (MULTI-BAND) ----
        with rasterio.open(image_path) as src:
            img = src.read()  # (C, H, W)
            img = np.transpose(img, (1, 2, 0))  # (H, W, C)

        # ---- NORMALIZE (same as training) ----
        img = img.astype(np.float32)
        for c in range(img.shape[-1]):
            band = img[..., c]
            min_val = band.min()
            max_val = band.max()
            if max_val - min_val > 0:
                img[..., c] = (band - min_val) / (max_val - min_val)
            else:
                img[..., c] = 0.0

        # ---- READ MASK ----
        with rasterio.open(mask_path) as src:
            mask = src.read(1)

        mask = (mask > 0).astype(np.uint8)

        # ---- PREDICTION ----
        predicted_mask = model.predict(np.expand_dims(img, axis=0))[0]

        # ---- THRESHOLD ----
        predicted_mask_thresh = (predicted_mask > 0.5).astype(np.uint8)

        # ---- SAVE ----
        save_image(predicted_mask, os.path.join(output_dir, f"prob_{file}"))
        save_image(predicted_mask_thresh, os.path.join(output_dir, f"pred_{file}"))

        # ---- METRICS ----
        dice = dice_coefficient(mask, predicted_mask_thresh)
        iou_score = iou(mask, predicted_mask_thresh)

        precision = precision_score(mask.flatten(), predicted_mask_thresh.flatten(), zero_division=0)
        recall = recall_score(mask.flatten(), predicted_mask_thresh.flatten(), zero_division=0)
        f1 = f1_score(mask.flatten(), predicted_mask_thresh.flatten(), zero_division=0)

        dice_scores.append(dice)
        iou_scores.append(iou_score)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        count += 1
        if number != -1 and count >= number:
            break

    # ---- AVERAGE METRICS ----
    mean_dice = np.mean(dice_scores)
    mean_iou = np.mean(iou_scores)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_f1 = np.mean(f1_scores)

    print(f"Mean Dice Coefficient: {mean_dice}")
    print(f"Mean IoU: {mean_iou}")
    print(f"Mean Precision: {mean_precision}")
    print(f"Mean Recall: {mean_recall}")
    print(f"Mean F1 Score: {mean_f1}")

    return mean_dice, mean_iou, mean_precision, mean_recall, mean_f1


from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

model_types = ['residual']   # or ['residual'], ['dense']

image_dir = './dataset/images'
mask_dir  = './dataset/masks'

train = True

# ---- LOAD DATA ----
images, masks = load_data_new(image_dir, mask_dir)

print(images.shape)
print(masks.shape)

# ---- SPLIT DATA ----
X_train, X_temp, y_train, y_temp = train_test_split(
    images, masks, test_size=0.3, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print("Train:", X_train.shape)
print("Val:", X_val.shape)
print("Test:", X_test.shape)

input_shape = X_train.shape[1:]

for model_type in model_types:

    model_name = f'UNet_{model_type}'
    print(model_name)

    # ---- SELECT MODEL ----
    if model_type == 'dense':
        model_fn = dense_unet

    elif model_type == 'residual':
        model_fn = residual_unet

    elif model_type == 'unet++':
        model_fn = lambda shape: dense_unet_plus_plus(shape, base_filters=32)

    # ---- TRAIN ----
    if train:
        model = model_fn(input_shape)

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=3,
            batch_size=1,
            callbacks=[early_stopping]
        )

        model.save(model_name)

    # ---- LOAD MODEL ----
    model = tf.keras.models.load_model(model_name, compile=False)
    model.compile()

    # ---- TEST EVALUATION ----
    output_dir = f'./output/{model_type}/'
    os.makedirs(output_dir, exist_ok=True)

    print("\nRunning on TEST set...")

    # Save test data temporarily for predict1
    test_img_dir = "./temp_test_images"
    test_mask_dir = "./temp_test_masks"

    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(test_mask_dir, exist_ok=True)

    import tifffile as tiff

    # Save test tiles to temp folders
    for i in range(len(X_test)):
        tiff.imwrite(os.path.join(test_img_dir, f"{i}.tif"), X_test[i])
        tiff.imwrite(os.path.join(test_mask_dir, f"{i}.tif"), y_test[i].squeeze())

    # ---- PREDICT ----
    predict1(
        model,
        test_mask_dir,
        test_img_dir,
        output_dir
    )