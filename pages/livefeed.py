import streamlit as st
import cv2
import requests
from PIL import Image, ImageFile
from utils import get_helmet_inference, get_numberplate_inference
from webapp import get_pdf_base64, get_location, generate_pdf_report, extract_number
from geopy.geocoders import Nominatim
from datetime import datetime
import numpy as np
from email_feature import send_email

ImageFile.LOAD_TRUNCATED_IMAGES = True


st.title("Live Feed")

frame_window = st.empty()    

location_coords = get_location()
current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
flag = True

if location_coords:
    latitude, longitude = location_coords
    geolocator = Nominatim(user_agent="crime_prediction_app")
    location = geolocator.reverse((latitude, longitude), language="en")
    location_name = location.address if location else "Unknown Location"
else:
    location_name = "Unknown Location"
curr_frame = st.empty()
captured_frame = st.empty()

webcam_input = st.camera_input('click a picture')

if webcam_input:
    with open('temp_file.jpg', "wb") as f:
        f.write(webcam_input.getvalue())


    opencv_image = cv2.imread('temp_file.jpg', 1)
    got_result_helmet = get_helmet_inference('temp_file.jpg')
    got_result_numberplate = get_numberplate_inference('temp_file.jpg')
    number_plate_text = extract_number('temp_file.jpg')

    faults = {}
    for i in got_result_helmet:
        faults[i['class']] = faults.get(i['class'], 0) + 1

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
    pdf_filename = generate_pdf_report(faults, number_plate_text, location_name, current_date, 'detected_image.jpg')
    pdf_base64 = get_pdf_base64(pdf_filename)
    st.markdown(f'<a href="data:application/octet-stream;base64,{pdf_base64}" download="helmet_tripling.pdf">Click here to download PDF</a>', unsafe_allow_html=True)        