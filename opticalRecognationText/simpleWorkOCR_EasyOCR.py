import pytesseract as pt
from easyocr import Reader
from opticalRecognationText.helper import straight_recognition, similarity_score

path_to_tesseract = r"../tesseract.exe"

pt.pytesseract.tesseract_cmd = (path_to_tesseract)
dataset = []
for i in range(1,3):
    dataset.append(f"./imgs/captcha/{i}.png")

image_path = "../imgs/captcha/2.png"

def test_recognition(rec_type, val_type):
    if rec_type == "straight":
        recognized_text = straight_recognition(image_path)
    elif rec_type == "easyOCR":
        reader = Reader(['en'])
        result = reader.readtext(image_path)
        recognized_text = " ".join([pair[1] for pair in result])


    ground_truth = "Ground truth text here"

    if val_type == "exact_match":
        is_match = (recognized_text == ground_truth)
    elif val_type == "similarity_score":
        similarity = similarity_score(recognized_text, ground_truth)
        is_match = (similarity > 0.8)

    # Write results to a file
    with open("recognition_results.txt", "a") as file:
        file.write(f"Recognized: {recognized_text}, Expected: {ground_truth}, Match: {is_match}\n")

    # Return statistical results
    return {
        "recognized_text": recognized_text,
        "ground_truth": ground_truth,
        "is_match": is_match,
        "similarity_score": similarity if val_type == "similarity_score" else None
    }
if __name__ == '__main__':
    result_straight_exact = test_recognition("straight", "exact_match")
    print("Straight Recognition - Exact Match:")
    print(result_straight_exact)

    # Example 2: EasyOCR recognition with similarity score validation
    result_easyOCR_similarity = test_recognition("easyOCR", "similarity_score")
    print("EasyOCR Recognition - Similarity Score:")
    print(result_easyOCR_similarity)