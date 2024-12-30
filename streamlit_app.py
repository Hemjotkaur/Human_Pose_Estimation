import time
import streamlit as st
from PIL import Image
import numpy as np
import cv2

DEMO_IMAGE = 'stand.jpg'

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]


width = 368
height = 368
inWidth = width
inHeight = height

net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

st.sidebar.title("Welcome👋 to Dynamic Pose Analyzer.")
st.sidebar.markdown("""
    ### 🤔 What is Human Pose Detection?
    Human pose detection is a computer vision task that involves identifying the positions of key points on the human body, such as joints and limbs. This technology is widely used in various applications, including:
    - **🏅 Sports Analysis**: To analyze athlete's movements.
    - **🏋️‍♂️ Fitness Apps**: To provide feedback on exercise form.
    - **🕶️ Augmented Reality**: For interactive experiences.
""")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🚀 About This App :-")

st.sidebar.markdown("""
   This application allows users to perform human pose detection on images and provides real-time pose detection experience using advanced machine learning techniques.
""")
st.sidebar.markdown("---")
st.sidebar.text("For any Query related issues contact at 📧-'hemjotkaur786@gmail.com'")

st.title("Dynamic Pose Analyzer 🤖")

st.text('Want to Detect the Pose for Your Image, here you go.... 🚀')

img_file_buffer = st.file_uploader("Upload an image, Make sure you have a clear image. ", type=[ "jpg", "jpeg",'png'])

if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))
    st.subheader('Original Image')
    st.image(
    image, caption=f"Original Image", use_container_width=True
) 

st.markdown("     ")
thres = st.slider('Choose threshold Value below :-',min_value = 0,value = 20, max_value = 100,step = 5)
st.text("Please select your desired threshold value. This threshold is responsible for detecting points in the pose estimation.")
st.markdown("---")
st.text(" To Process your image, please click the button below :-")
thres = thres/100

@st.cache_data
def poseDetector(frame):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    
    out = net.forward()
    out = out[:, :19, :, :]
    
    assert(len(BODY_PARTS) == out.shape[1])
    
    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]

        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > thres else None)
        
        
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                
    return frame
if st.button("Process Image"):
    with st.spinner("Estimating your Image..."):
      time.sleep(2)
    output = poseDetector(image)
    st.subheader('Positions Estimated')
    st.image(output, caption="Positions Estimated", use_container_width=True)

    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

   
    _, buffer = cv2.imencode('.png', output_rgb) 
    img_bytes = buffer.tobytes()  

    st.download_button(
        label="Download Processed Image",
        data=img_bytes,
        file_name="processed_image.png",
        mime="image/png"
    )    
   
st.markdown("---")

def pose_estimation(cap):

    while(cap.isOpened()):
        
        ret, frame = cap.read()
        frame = cv2.resize(frame,(0,0),fx = 1.0,fy = 1.0)
        frame = cv2.flip(frame, 1)
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
    
        net.setInput(cv2.dnn.blobFromImage(frame, 2.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    
        out = net.forward()
        out = out[:, :19, :, :]
    
        assert(len(BODY_PARTS) == out.shape[1])
    
        points = []
        for i in range(len(BODY_PARTS)):
            heatMap = out[0, i, :, :]

            _, conf, _, point = cv2.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]
            points.append((int(x), int(y)) if conf > thres else None)
            
        for pair in POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            assert(partFrom in BODY_PARTS)
            assert(partTo in BODY_PARTS)

            idFrom = BODY_PARTS[partFrom]
            idTo = BODY_PARTS[partTo]
            
            if points[idFrom] and points[idTo]:
                cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                
        t, _ = net.getPerfProfile()
        img = frame
        
        
        size = (frameWidth, frameHeight) 
        
        cv2.imshow('pose',frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

    cv2.destroyAllWindows()
    
cv2.waitKey(0)
cv2.destroyAllWindows()
st.subheader("If you would like to experience real-time pose detection, please click the button below.")
if st.button("Enable Web Cam"):
     cap = cv2.VideoCapture(0) 
     pose_estimation(cap)

st.markdown("---")
st.title("How to use the App for Better Results-")
st.text("✔️ Use images with good lighting and contrast.")
st.text("✔️ Avoid occlusions by ensuring all body parts are visible and not blocked.")
st.text("✔️ For videos, ensure a steady camera to capture clear frames.")
st.text("✔️ Stop the video by pressing the key q on keyboard.")            

st.markdown("---")
st.markdown("### ! Thank You for Using the Human Pose Estimation App! ")
st.markdown("    ")
st.markdown("""
    I hope you found this app helpful and informative. 
    Keep exploring the fascinating world of human pose detection.
    If you have any feedback or suggestions, feel free to reach out.
""")

