from inference_sdk import InferenceHTTPClient
import cv2
import easyocr
ALPR_CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="lf8JQqg2UBMk52pV9Scq"
)

HELMET_CLIENT = InferenceHTTPClient(
    api_url = 'https://detect.roboflow.com',
    api_key = 'lf8JQqg2UBMk52pV9Scq'
)

def extract_plate(img_path):
    img = cv2.imread(img_path)
    plate_img = img.copy()
    plate_cascade = cv2.CascadeClassifier('./indian_license_plate.xml')
    plate_rect = plate_cascade.detectMultiScale(plate_img, scaleFactor = 1.3, minNeighbors = 7)
    print(plate_rect)
    return plate_rect[0]

def get_helmet_inference(image_path):
    result = HELMET_CLIENT.infer(image_path, model_id="violation-detection-xsjs0/1")
    print(result)
    return result['predictions']

def get_numberplate_inference(image_path):
    result = ALPR_CLIENT.infer(image_path, model_id="alpr-yolov8/4")
    print(result)
    return result
    
def extract_number(image_path):
    result = extract_plate(image_path)
    x1, y1 = int(result[0]), int(result[1])
    x2 , y2 = x1+int(result[2]), y1+int(result[3])
    image = cv2.imread(image_path, 1)
    number_plate_image = image[y1:y2, x1:x2]
    reader = easyocr.Reader(['en'])
    ocr_result = reader.readtext(number_plate_image)
    number_plate_text = " ".join([result[1] for result in ocr_result])
    return number_plate_text

if __name__ == '__main__':
    result = extract_number('temp_file.jpg')
    print(result)