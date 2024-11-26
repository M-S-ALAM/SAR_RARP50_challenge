import streamlit as st
import cv2
import tempfile
import os
from extract_frame import extract_frames

st.title('Semantic Segmentation of Surgical Tools')

# Create a file uploader to accept video files
uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])

def process_frame(frame):
    # Apply your processing here
    # For demonstration, let's just pass the frame as is
    return frame

if uploaded_file is not None:
    # Use tempfile to create a temporary file path for the uploaded video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    tfile.close()

    # Load the video using OpenCV
    cap = cv2.VideoCapture(tfile.name)

    # Prepare to save the processed video
    # Define the codec and create VideoWriter object
    if cap.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID' for .avi format
        out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
        st.success("Video successfully loaded.")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Process the frame
                processed_frame = process_frame(frame)

                # Write the processed frame to the output video
                out.write(processed_frame)

                # To display the frame, convert it to RGB (OpenCV uses BGR by default)
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, channels="RGB", use_column_width=True)

        finally:
            # Release everything when job is finished
            cap.release()
            out.release()
            os.unlink(tfile.name)  # Optionally delete the tempfile
            st.info("Finished processing video.")

        # Extract frames from the saved video
        extract_frames('output.mp4', '/home/shobot/Shahbaz_project/SAR_RARP50_challenge/Database/Testing/frame', 10)
        st.success("successfully extraction done!.")
    else:
        st.error("Error loading video.")
else:
    st.warning("Please upload a video file.")
