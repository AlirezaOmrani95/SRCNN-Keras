

from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import load_img
import PIL
from tensorflow.keras import backend as K
plt.switch_backend('WebAgg')



class OxfordPets(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths):
        self.batch_size = batch_size
        self.img_size_out = img_size
        self.input_img_paths = input_img_paths

    def __len__(self):
        return len(self.input_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]

        x = np.zeros((self.batch_size,) + self.img_size_out + (1,), dtype="float32")
        y = np.zeros((self.batch_size,) + self.img_size_out + (1,), dtype="float32")

        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size_out)
            img_out = np.array(img)
            img_out = cv2.cvtColor(img_out,cv2.COLOR_RGB2YCrCb)
            img_out = img_out / 255
            img_out = np.expand_dims(img_out [:,:,0],-1)
            img_in = get_lowres_image(img,2)
            img_in = cv2.cvtColor(np.array(img_in),cv2.COLOR_RGB2YCrCb)
            img_in = img_in / 255
            img_in = np.expand_dims(img_in [:,:,0],-1)
            x[j] = img_in
            y[j] = img_out
        return x, y

def get_lowres_image(img, upscale_factor):
    """Return low-resolution image to use as model input."""
    img_ =  img.resize(
        (img.size[0] // upscale_factor, img.size[1] // upscale_factor),
        PIL.Image.BICUBIC)
    return cv2.resize(np.array(img_),(img.size[0],img.size[1]),interpolation=cv2.INTER_CUBIC)

def model():

    SRCNN = keras.models.Sequential()
    SRCNN.add(keras.layers.Conv2D(filters=128, kernel_size = (9, 9), kernel_initializer='glorot_uniform',
                         activation='relu', padding='same', use_bias=True, input_shape=(None, None, 1))) 
    SRCNN.add(keras.layers.Conv2D(filters=64, kernel_size = (3, 3), kernel_initializer='glorot_uniform',
                         activation='relu', padding='same', use_bias=True))
    SRCNN.add(keras.layers.Conv2D(filters=1, kernel_size = (5, 5), kernel_initializer='glorot_uniform',
                         activation='linear', padding='same', use_bias=True))

    SRCNN.compile(optimizer=keras.optimizers.Adam(), loss='mse')
    return SRCNN

SR_Path = '/media/aro/New Volume/Super resolution dataset/set5/Train/'
Valid_path = '/media/aro/New Volume/Super resolution dataset/set5/Valid/'

input_SR_paths = [
        os.path.join(SR_Path, fname)
        for fname in os.listdir(SR_Path)
        if fname.endswith(".png")
    ]

Valid_SR_paths = [
        os.path.join(Valid_path, fname)
        for fname in os.listdir(Valid_path)
        if fname.endswith(".png")
    ]


model = model()
img_size = (512,512)
train_bsize = 2

train_generator = OxfordPets(batch_size = train_bsize, img_size = img_size, input_img_paths = input_SR_paths)
Valid_generator = OxfordPets(batch_size = train_bsize, img_size = img_size, input_img_paths = Valid_SR_paths)

save = ModelCheckpoint('/media/aro/New Volume/Super resolution dataset/set5/results/SRCNN/laplacian 500/laplacian SRCNN.hdf5', save_best_only=True)

if os.path.exists('/media/aro/New Volume/Super resolution dataset/set5/results/SRCNN/laplacian 500/laplacian SRCNN.hdf5'):
  model.load_weights('/media/aro/New Volume/Super resolution dataset/set5/results/SRCNN/laplacian 500/laplacian SRCNN.hdf5')

history = model.fit(train_generator,epochs=50,callbacks=[save], validation_data=Valid_generator)

plt.plot(history.history['loss'],label = "Training")
plt.plot(history.history['val_loss'],label = "Validation")
plt.title("weighted MSE loss trend")
plt.ylabel("MSE Value")
plt.xlabel("No. epoch")
plt.legend(loc = "upper left")
plt.show()

s_input = load_img("/media/aro/New Volume/Super resolution dataset/set5/results/input/ppt3.png", target_size=(640,512))

s = get_lowres_image(s_input,2)
s = cv2.cvtColor(np.array(s),cv2.COLOR_RGB2YCrCb)
S_Y = s[:,:,0]
_max = np.max(S_Y)
_min = np.min(S_Y)
S_Y_ex = np.expand_dims(S_Y,-1)
S_Y_ex = np.expand_dims(S_Y_ex,0)
a = model.predict(S_Y_ex)
a = np.squeeze(a)
a_min = np.min(a)
a = a+np.abs(a_min)
a_max = np.max(a)
a = (a/a_max)*255
a = np.uint8(a)
S_Y =S_Y+(a*0.1)
a = (S_Y/np.max(S_Y))*255
cv2.imwrite('/media/aro/New Volume/Super resolution dataset/set5/results/SRCNN/laplacian 500/ppt3_out_luminance_new1.png',a)
