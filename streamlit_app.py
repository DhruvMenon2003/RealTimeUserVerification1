import streamlit as st
import cv2
import os
from deepface import DeepFace
from PIL import Image

# Temporary storage for uploaded image
TEMP_IMAGE_PATH = "temp_uploaded_image.jpg"

# Streamlit UI
st.title("Real-Time Face Verification")

# Verify button
if st.button("Verify"):
    # Upload image
    uploaded_file = st.file_uploader("Upload a picture", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Save the uploaded image temporarily
        with open(TEMP_IMAGE_PATH, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Start real-time face verification
        st.write("Starting real-time face verification...")

        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Unable to access the camera.")
        else:
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame.")
                    break

                # Display the live video feed
                cv2.imshow("Real-Time Face Verification", frame)

                # Verify the face in the frame with the uploaded image
                try:
                    result = DeepFace.verify(frame, TEMP_IMAGE_PATH, model_name="FaceNet512", enforce_detection=False)
                    if result["verified"]:
                        st.success("Verified! ✅")
                        break
                    else:
                        st.error("Verification Failed! ❌")
                        break
                except Exception as e:
                    st.error(f"Error during verification: {e}")

                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Release the camera and close OpenCV windows
            cap.release()
            cv2.destroyAllWindows()

        # Remove the temporary image file
        os.remove(TEMP_IMAGE_PATH)
