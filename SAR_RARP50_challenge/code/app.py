# Import necessary libraries
import streamlit as st
import cv2
import tempfile
import os
from extract_frame import VideoProcessor

# Title for the Streamlit app
st.title('Semantic Segmentation of Surgical Tools')

# File uploader to accept video files from the user
uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])


def process_frame(frame):
    """
    Process a single video frame.

    Args:
        frame (numpy.ndarray): Input video frame.

    Returns:
        numpy.ndarray: Processed video frame.
    """
    # Apply any desired processing to the frame
    # For demonstration, the frame is returned as-is
    return frame


if uploaded_file is not None:
    # Create a temporary file to save the uploaded video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    tfile.close()

    # Load the uploaded video using OpenCV
    cap = cv2.VideoCapture(tfile.name)

    # Check if the video was successfully loaded
    if cap.isOpened():
        st.success("Video successfully loaded.")

        # Prepare to save the processed video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
        out = cv2.VideoWriter(
            'output.mp4',
            fourcc,
            20.0,
            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        )

        try:
            # Process each frame in the video
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Apply processing to the frame
                processed_frame = process_frame(frame)

                # Write the processed frame to the output video
                out.write(processed_frame)

                # Convert the frame to RGB format for Streamlit display (OpenCV uses BGR)
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, channels="RGB", use_column_width=True)
        finally:
            # Release resources after processing
            cap.release()
            out.release()
            os.unlink(tfile.name)  # Delete the temporary file
            st.info("Finished processing video.")

        # Extract frames from the processed video
        extract_frames(
            'output.mp4',
            '/home/shobot/Shahbaz_project/SAR_RARP50_challenge/Database/Testing/frame',
            10
        )
        st.success("Frame extraction completed successfully!")
    else:
        # Display error message if the video could not be loaded
        st.error("Error loading video.")
else:
    # Warning message when no file is uploaded
    st.warning("Please upload a video file.")
