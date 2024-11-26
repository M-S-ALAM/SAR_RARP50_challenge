import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
def preprocess_image(image_path, target_size):
    # Load the image file, ensuring it's resized to match the model's expected input
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)  # Convert the image to an array
    img_array = img_array / 255.0  # Scale the image to the [0, 1] range
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


def predict_and_display(image_path, Actual_image):
    # Assume the model expects images of size 256x256
    img_array = preprocess_image(image_path, target_size=(256, 256))
    actual_array = preprocess_image(Actual_image, target_size=(256, 256))

    # Predict the output using the loaded model
   # pred = model.predict(img_array)

    # Display the original image and the prediction
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(tf.keras.preprocessing.image.array_to_img(img_array[0]))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(tf.keras.preprocessing.image.array_to_img(actual_array[0]))
    axes[1].set_title('Actual Image')
    axes[1].axis('off')

    plt.show()


def main():
    data = pd.read_csv('/home/shobot/Shahbaz_project/SAR_RARP50_challenge/Database/test_data.csv')
    for i in range(5):
      j = np.random.randint(0, len(data))
      image_path = data.iloc[j]['Frame']
      actual_path = data.iloc[j]['Segmentation']
      predict_and_display(image_path, actual_path)


if __name__=='__main__':
    main()