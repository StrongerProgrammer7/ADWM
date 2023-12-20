
from helper import *
import pytesseract as pt
from easyocr import Reader

path_to_tesseract = r"../tesseract.exe"
pt.pytesseract.tesseract_cmd = (path_to_tesseract)
imgs_path = "capcha_2"

def evaluate_dataset(dataset, rec_type, val_type,typeText):
    results = []
    for image_path, ground_truth in dataset:
        result = test_recognition(rec_type, val_type, image_path, ground_truth,typeText)
        results.append(result)
    return results

def evaluate_dataset_one_result(dataset, rec_type, val_type,typeText=None):
    results = []
    similarity_score = 0
    bestRes = {}
    for image_path, ground_truth in dataset:
        result = test_recognition(rec_type, val_type, image_path, ground_truth,typeText)
        if(result['similarity_score'] > similarity_score):
            similarity_score =  result['similarity_score'];
            bestRes = result
    results.append(bestRes)
    return results

def test_recognition(rec_type, val_type, image_path, ground_truth,typeText):
    if rec_type == "straight":
        recognized_text = straight_recognition(image_path)
    elif rec_type == "straight_deleteSpecialSymbol_tolower":
        recognized_text = straight_recognition(image_path)
        recognized_text = ''.join(e for e in recognized_text if e.isalnum()).lower()
        ground_truth = ''.join(e for e in ground_truth if e.isalnum()).lower()
    elif rec_type == "easyOCR":
        reader = Reader(['en'])
        result = reader.readtext(image_path)
        recognized_text = " ".join([pair[1] for pair in result])

    if val_type == "exact_match":
        is_match = (recognized_text == ground_truth)
    elif val_type == "similarity_score":
        similarity = similarity_score(recognized_text, ground_truth)
        is_match = (similarity > 0.8)

    if(typeText != None):
        angle = '-'
        if(len(image_path.split('_')) > 2):
            angle = image_path.split('_')[3].split('.')[0]
        #print(angle)
        with open(f"recognition_results_{typeText}.txt", "a",encoding="utf8") as file:
            file.write(f"Recognized: {recognized_text},angle: {angle}; Expected: {ground_truth}, Match: {is_match}\n")

    return {
        "recognized_text": recognized_text,
        "ground_truth": ground_truth,
        "is_match": is_match,
        "similarity_score": similarity if val_type == "similarity_score" else None
    }
if __name__ == '__main__':
    original_dataset = load_dataset("imgs\capcha_2")
    dataset = augment_dataset(original_dataset)
    results = evaluate_dataset(dataset,"straight_deleteSpecialSymbol_tolower","similarity_score",2)

    results_od_one = evaluate_dataset_one_result(original_dataset, "straight", "similarity_score", None)
    # results_od = evaluate_dataset(original_dataset,"straight","similarity_score",3)
    # results_od_lower = evaluate_dataset(original_dataset, "straight_deleteSpecialSymbol_tolower", "similarity_score", 4)

    results_d_one = evaluate_dataset_one_result(dataset, "straight", "similarity_score", None)
    # results_d = evaluate_dataset(dataset, "straight", "similarity_score", 1)
    # results_d_lower = evaluate_dataset(dataset, "straight_deleteSpecialSymbol_tolower", "similarity_score", 2)
    print(results_od_one)
    print(results_d_one)

    #OCR
    results_od = evaluate_dataset(original_dataset, "easyOCR", "similarity_score", 'OCR1')

    results_d = evaluate_dataset(dataset, "easyOCR", "similarity_score", "OCR2")
