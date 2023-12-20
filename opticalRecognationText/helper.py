import os
from difflib import SequenceMatcher
import pytesseract as pt
from PIL import Image
from turtle import pd

def load_dataset(dataset_path):
    dataset = []
    #print(os.listdir("/High-level prommaing/Python/ADWM/imgs/capcha_2"))
    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg"):  # Assuming your images are in JPG format
            image_path = os.path.join(dataset_path, filename)

            # Assuming corresponding ground truth text files have the same name but end with ".txt"
            text_filename = os.path.splitext(filename)[0] + ".txt"
            text_path = os.path.join(dataset_path, text_filename)

            with open(text_path, "r",encoding='utf8') as text_file:
                ground_truth = text_file.read().strip()
            #print(ground_truth)
            dataset.append((image_path, ground_truth))

    return dataset

def rotate_image(image_path, angle,ind):
    img = Image.open(image_path)
    #print(img.filename)
    rotated_img = img.rotate(angle)
    rotated_image_path = f"../imgs/capcha_2/rotate/rotated_{ind}_{angle}.jpg"
    rotated_img.save(rotated_image_path)
    #print(rotated_img)
    return rotated_image_path

def augment_dataset(dataset):
    augmented_dataset = []
    ind = 0
    for image_path, ground_truth in dataset:
        for angle in range(-20, 21):
            #print(angle)
            rotated_image_path = rotate_image(image_path, angle,ind)
            augmented_dataset.append((rotated_image_path, ground_truth))
        ind+=1
    return augmented_dataset

def save_dataset_to_excel(dataset, output_excel_path):
    df = pd.DataFrame(dataset, columns=["Image Path", "Ground Truth"])
    df.to_excel(output_excel_path, index=False)

def straight_recognition(image_path):
    return pt.image_to_string(image_path,lang='rus+eng')

def similarity_score(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()