import streamlit as st
import cv2
from PIL import Image
import numpy as np
from utils import get_helmet_inference, get_numberplate_inference
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.platypus import Image as pdf_image
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime
import requests
from geopy.geocoders import Nominatim
import os
import base64
from email_feature import send_email
from random import randint, randrange
import easyocr
import pandas as pd
import random
from ultralytics import YOLO


emails = list(pd.read_csv('emails.csv')['emails'])
stored_numbers = list(pd.read_csv('emails.csv')['numberplate'])

tripling_punishment = 'In Andhra Pradesh, the penalty for triple riding on a two-vehicle is 1,000/- for the first offense, 500/- for the second offense, and 1,500/- for repeat offenses'

no_helmet_punishment = 'In India, the fine for riding a bike without a helmet is 1,000/- under section 194D of the Motor Vehicles Act, 2019. You can pay this fine either online or offline'

def get_pdf_base64(pdf_filename):
    with open(pdf_filename, "rb") as f:
        pdf_file = f.read()
    return base64.b64encode(pdf_file).decode('utf-8')


def get_location():
    response = requests.get("https://ipinfo.io")
    data = response.json()
    if "loc" in data:
        latitude, longitude = map(float, data["loc"].split(","))
        return latitude, longitude
    else:
        return None

def generate_pdf_report(faults, number_plate,location, date, image_path):
    filename = "helmet_tripling.pdf"
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    Story = []


    title = Paragraph("Helmet and Triple Riding Detection", styles['Title'])
    Story.append(title)
    Story.append(Spacer(1, 0.2 * inch))
    fault_string = " ".join(faults)
    major_cat_text = Paragraph(f"<b>Faults Found:</b> {fault_string}<br/><br/><b>Possible punishment:</b> {tripling_punishment} and {no_helmet_punishment}")
    Story.append(major_cat_text)
    Story.append(Spacer(1, 0.2 * inch))

    additional_info = Paragraph(f"<b>Date:</b> {date}<br/>"
                                f"<b>Location:</b> {location}<br/><br/>"
                                f"The Number Plate value: {number_plate}<br/><br/>"
                                "The Crime was identified by our YOLOV8 model and it has higher accuracy rate so please inform regulatory authority as soon as you get this report.",
                                styles['Normal'])
    Story.append(additional_info)
    Story.append(Spacer(1, 0.2 * inch))

    # Image
    if os.path.exists(image_path):
        im = pdf_image(image_path, 4 * inch, 3 * inch)
        Story.append(im)
        Story.append(Spacer(1, 0.2 * inch))

    doc.build(Story)
    return filename

def extract_plate(img):
    model = YOLO('indianlicenseplate.pt')
    model.cpu()
    results = model(img)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        return boxes

def extract_number(image_path):
    try:
        result = extract_plate(image_path)[0]
        print(result)
        x1, y1 = int(result[0]), int(result[1])
        x2 , y2 = x1+int(result[2]), y1+int(result[3])
        print(f"X1 - {x1}, X2 - {x2}, Y1 - {y1}, Y2 - {y2}")
        image = cv2.imread(image_path, 1)
        number_plate_image = image[y1:y2, x1:x2]
        reader = easyocr.Reader(['en'])
        ocr_result = reader.readtext(number_plate_image)
        number_plate_text = " ".join([result[1] for result in ocr_result])
        print(number_plate_text)
        return number_plate_text
    except:
        print("Numberplate not found")




pdf_base64 = None
# Display title
st.title("Helmet and Triple Riding Detection:-")

# File uploader widget
file = st.file_uploader("Upload Image: ", type=['jpg', 'mp4'])
trigger = []
flag = 0
if (file is not None):
    if file.name.split('.')[-1] == 'jpg':
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
    
    if file.name.split('.')[-1] == 'mp4':
        with open('temp_video.mp4', "wb") as f:
            f.write(file.getbuffer())
    


    location_coords = get_location()
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    if location_coords:
        latitude, longitude = location_coords
        geolocator = Nominatim(user_agent="crime_prediction_app")
        location = geolocator.reverse((latitude, longitude), language="en")
        location_name = location.address if location else "Unknown Location"
    else:
        location_name = "Unknown Location"



    if st.button('Check'):
        print(file.name)
        if file.name.split('.')[-1] == 'jpg' or file.name.split('.')[-1] == 'jpeg':
            with open('temp_file.jpg', "wb") as f:
                f.write(file.getbuffer())
            opencv_image = cv2.imread('temp_file.jpg', 1)
            got_result_helmet = get_helmet_inference('temp_file.jpg')
            # got_result_numberplate = get_numberplate_inference('temp_file.jpg')
            number_plate_text = extract_number('temp_file.jpg')

        else:
            video = cv2.VideoCapture('temp_video.mp4')
            while video.isOpened():
                ret, frame = video.read()
                val = randrange(100, 500)
                if ret == True:
                    if flag == 1 and trigger.count(1) == val:
                        cv2.imwrite('temp_file.jpg', frame)
                        break
                    flag = randint(0, 1)
                    trigger.append(flag)
                    print(flag)
        opencv_image = cv2.imread('temp_file.jpg', 1)
        got_result_helmet = get_helmet_inference('temp_file.jpg')
            # got_result_numberplate = get_numberplate_inference('temp_file.jpg')
        number_plate_text = extract_number('temp_file.jpg')


        faults = {}
        for i in got_result_helmet:
            faults[i['class']] = faults.get(i['class'], 0) + 1


        print(faults)
        print(number_plate_text)
        for detection in got_result_helmet:
            x, y, width, height = detection['x'], detection['y'], detection['width'], detection['height']
            start_point = (int(x), int(y))
            end_point = (int(x + width), int(y + height))
            color = (255, 0, 0)
            thickness = 2
            cv2.rectangle(opencv_image, start_point, end_point, color, thickness)
                    
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(opencv_image, f"{detection['class']} ({detection['confidence']:.2f})", (int(x), int(y - 10)), font, 0.5, (255, 255, 255), 2)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(opencv_image)
        pil_image.save('detected_image.jpg')
        st.image(pil_image, caption="Processed Image")
        pdf_filename = generate_pdf_report(faults, number_plate_text,location_name, current_date, 'detected_image.jpg')
        pdf_base64 = get_pdf_base64(pdf_filename)
        get_email = 'kallepalliteerdhanadh56@gmail.com'
        print(get_email)
        if len(faults) > 0:
            try:
                if len(number_plate_text) > 4:
                    for i, number in enumerate(stored_numbers):
                        if str(number).find(number_plate_text[-3:-1]) != -1:
                            get_email = emails[i]
                            break
                else:
                    send_email(get_email, "sending as default", "Traffic Rules Broken", ["helmet_tripling.pdf"])
                st.markdown(f'<a href="data:application/octet-stream;base64,{pdf_base64}" download="helmet_tripling.pdf">Click here to download PDF</a>', unsafe_allow_html=True)
                st.write(f'mail sending to {get_email}')
                send_email(get_email, "Traffic Rules Broken", "Find attachment for the details we got below.", ["helmet_tripling.pdf"])
            except: 
                send_email(get_email, "sending as default", "Traffic Rules Broken", ["helmet_tripling.pdf"])

        else:
            send_email(get_email, "sending as default", "Traffic Rules Broken", ["helmet_tripling.pdf"])

