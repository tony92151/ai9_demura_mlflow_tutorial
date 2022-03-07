import tensorflow as tf
import tensorflow_addons as tfa
import cv2

class wei_augumentation(object):
    def __call__(self, img):
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.rgb_to_grayscale(img)
        img2 = tf.image.sobel_edges(img[None, ...])
        equal_img = tfa.image.equalize(img, bins=256)
        img = tf.concat([equal_img, img2[0, :, :, 0]], 2)
        image_array = tf.keras.preprocessing.image.array_to_img(img)

        return image_array

    def __repr__(self):
        return self.__class__.__name__ + '()'


# In[12]:


class CLAHE(object):
    def __call__(self, img):
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'
