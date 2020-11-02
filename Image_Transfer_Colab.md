# Image_Transfer 과제  
### 1. 설정
```
import functools
import os

from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

print("TF Version: ", tf.__version__)
print("TF-Hub version: ", hub.__version__)
print("Eager mode enabled: ", tf.executing_eagerly())
print("GPU available: ", tf.test.is_gpu_available())
```
### 2. 이미지 로딩 및 시각화 기능 정의
```
def crop_center(image):
  """Returns a cropped square image."""
  shape = image.shape
  new_shape = min(shape[1], shape[2])
  offset_y = max(shape[1] - shape[2], 0) // 2
  offset_x = max(shape[2] - shape[1], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  return image

@functools.lru_cache(maxsize=None)
def load_image(image_url, image_size=(256, 256), preserve_aspect_ratio=True):
  """Loads and preprocesses images."""
  # Cache image file locally.
  image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)
  # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
  img = plt.imread(image_path).astype(np.float32)[np.newaxis, ...]
  if img.max() > 1.0:
    img = img / 255.
  if len(img.shape) == 3:
    img = tf.stack([img, img, img], axis=-1)
  img = crop_center(img)
  img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
  return img

def show_n(images, titles=('',)):
  n = len(images)
  image_sizes = [image.shape[1] for image in images]
  w = (image_sizes[0] * 6) // 320
  plt.figure(figsize=(w  * n, w))
  gs = gridspec.GridSpec(1, n, width_ratios=image_sizes)
  for i in range(n):
    plt.subplot(gs[i])
    plt.imshow(images[i][0], aspect='equal')
    plt.axis('off')
    plt.title(titles[i] if len(titles) > i else '')
  plt.show()
```
### 3. 이미지 예제 불러오기

```
# @title Load example images  { display-mode: "form" }

content_image_url = 'https://postfiles.pstatic.net/MjAyMDExMDJfMjA3/MDAxNjA0MjgzNzg0MDg4.RekVUHTuJTD3Ht3YGhes1xvVvyILI1f5ReopVut527Eg.nu6m-RL7Pi30P1K12QoYkq3Ipjg7yIiU_fBYwr0IscUg.JPEG.ansanghyun20/1.jpg?type=w773'  # @param {type:"string"}
style_image_url = 'https://m.damcoart.com/web/product/big/201611/1754_shop1_228308.jpg'  # @param {type:"string"}
output_image_size = 384  # @param {type:"integer"}

# The content image size can be arbitrary.
content_img_size = (output_image_size, output_image_size)
# The style prediction model was trained with image size 256 and it's the 
# recommended image size for the style image (though, other sizes work as 
# well but will lead to different results).
style_img_size = (256, 256)  # Recommended to keep it at 256.

content_image = load_image(content_image_url, content_img_size)
style_image = load_image(style_image_url, style_img_size)
style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')
show_n([content_image, style_image], ['Content image', 'Style image'])
```


### <span style="color:red">3번 수행결과</span>
![1](https://user-images.githubusercontent.com/62547169/97862141-3214b200-1d48-11eb-9fce-5769b5dc06ae.PNG)



### 4. TF-Hub 모듈 가져 오기
```
hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(hub_handle)
```

### 5. 이미지 스타일 화 시연
```
outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
stylized_image = outputs[0]
```
```
show_n([content_image, style_image, stylized_image], titles=['Original content image', 'Style image', 'Stylized image'])
```
### 5번 수행결과
![2](https://user-images.githubusercontent.com/62547169/97862482-c848d800-1d48-11eb-9760-6e2899eecd73.PNG)
