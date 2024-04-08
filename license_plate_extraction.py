import cv2
from ultralytics import YOLO
# def extract_plate(img):
# 	plate_img = img.copy()
# 	plate_cascade = cv2.CascadeClassifier('./indian_license_plate.xml')
# 	plate_rect = plate_cascade.detectMultiScale(plate_img, scaleFactor = 1.3, minNeighbors = 7)

# 	for (x,y,w,h) in plate_rect:
# 		a,b = (int(0.02*img.shape[0]), int(0.025*img.shape[1]))
# 		plate = plate_img[y+a:y+h-a, x+b:x+w-b, :]
		
# 		cv2.rectangle(plate_img, (x,y), (x+w, y+h), (51,51,255), 3)
# 		cv2.imshow('img',plate_img)
# 		cv2.waitKey(0)
# 		cv2.destroyAllWindows()
# 	return x, y, w, h

def extract_plate(img):
    model = YOLO('indianbikeplate.pt').load('indianbikeplate.pt')
    model.cpu()
    results = model(img)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        return boxes


if __name__ == '__main__':
    img = cv2.imread('temp_file.jpg')
    x, y, w, h = extract_plate(img)
    print(x, y, w, h)