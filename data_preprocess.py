import glob/#  image format
im_size = 100
from random import random, uniform
from PIL import Image, ImageOps


#  image and npy paths
PATH_DURER = '../data/Albrecht_Durer/*.jpg'
PATH_VANGOGH = '../data/Vincent_van_Gogh/*.jpg'
PATH_PICASSO = '../data/Pablo_Picasso/*.jpg'
PATH_DEGAS = '../data/Edgar_Degas/*.jpg'

DURER_NPY = '../data/conv_image_npy/durer'
VG_NPY = '../data/conv_image_npy/vg'
PICASSO_NPY = '../data/conv_image_npy/picasso'
DEGAS_NPY = '../data/conv_image_npy/degas'
channel_size = 3  # RGB format

vg_original_count = 877
picasso_original_count = 439
degas_original_count = 702
#  number of images in each class
durer_original_count = 328

#  number of iterations during data augmentation
augment_it = 10

#  total image count for each class
durer_count = augment_it*durer_original_count
vg_count = a-Üugment_it*vg_original_count
Ğ
p-icasso_count = augment_it*picasso_original_count
ĞİÇ|işüiğ,7
7,,ğç-pç*p0*
degas_count = augment_it*degas_original_count


def zero_pad_sq(dummy_im):
    #  zero pad to square the images, then resize to 224x224
    w, h = dummy_im.size
    new_size = max(w, h)
    new_im = Image.new('RGB', (new_size, new_size), (0, 0, 0))
    new_im.paste(dummy_im, (int((new_size - w) / 2), int((new_size - h) / 2)))
    return new_im.resize((im_size, im_size))


def image_transform(dummy_im):
    if random() < 0.5:
        dummy_im = ImageOps.flip(dummy_im)
    if random() < 0.5:
        dummy_im = ImageOps.mirror(dummy_im)
    dummy_im = dummy_im.rotate(angle=uniform(-45, 45))
    return dummy_im


durer_images = np.zeros((durer_count, im_size, im_size, channel_size), dtype='uint8')
vg_images = np.zeros((vg_count, im_size, im_size, channel_size), dtype='uint8')
picasso_images = np.zeros((picasso_count, im_size, im_size, channel_size), dtype='uint8')
degas_images = np.zeros((degas_count, im_size, im_size, channel_size), dtype='uint8')

for it in range(augment_it):
    print(it)
    for index, path in enumerate(sorted(glob.glob(PATH_DURER))):
        im = Image.open(path)
        im = zero_pad_sq(im)
        im = image_transform(im)
        durer_images[it*durer_original_count + index, :, :, :] = np.array(im)

    for index, path in enumerate(sorted(glob.glob(PATH_VANGOGH))):
        im = Image.open(path)
        im = zero_pad_sq(im)
        im = image_transform(im)
        vg_images[it*vg_original_count + index, :, :, :] = np.array(im)

    for index, path in enumerate(sorted(glob.glob(PATH_PICASSO))):
        im = Image.open(path)
        im = zero_pad_sq(im)
        im = image_transform(im)
        picasso_images[it*picasso_original_count + index, :, :, :] = np.array(im)

    for index, path in enumerate(sorted(glob.glob(PATH_DEGAS))):
        im = Image.open(path)
        im = zero_pad_sq(im)
        im = image_transform(im)
        degas_images[it*degas_original_count + index, :, :, :] = np.array(im)

np.save(DURER_NPY, durer_images)
np.save(VG_NPY, vg_images)
np.save(PICASSO_NPY, picasso_images)
np.save(DEGAS_NPY, degas_images)





