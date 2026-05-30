# ui.py
import cv2
import streamlit as st

from model import ActivityModel, CrowdGroupModel
from camera import (
    open_laptop_camera,
    open_iriun_camera,
    open_video_file,
    save_uploaded_to_temp,
)

st.set_page_config(page_title="Crowd Movement Analysis", layout="wide")
st.title("Crowd Movement Analysis using HAR & Crowd Group Detection")

# ---- Session state flags ----
for key in ["run_iriun", "run_laptop", "run_upload", "upload_path"]:
    if key not in st.session_state:
        st.session_state[key] = False if key != "upload_path" else None


def run_stream(source_type: str, cap):
    """
    For a given cv2.VideoCapture, run:
    Left  : ActivityModel (sit/stand/walk/run)
    Right : CrowdGroupModel (groups + heatmap)
    """
    if not cap or not cap.isOpened():
        st.error("Could not open video/camera.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    activity_model = ActivityModel(fps=fps)
    crowd_model = CrowdGroupModel(width=w, height=h, fps=fps)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Activity Detection ")
        placeholder_act = st.empty()
    with col2:
        st.subheader("Crowd Detection + Heatmap")
        placeholder_crowd = st.empty()

    flag_name = f"run_{source_type}"

    while st.session_state.get(flag_name, False):
        success, frame = cap.read()
        if not success:
            st.warning("No more frames / camera disconnected.")
            break

        frame_bgr = frame.copy()

        # Model 1: activity only
        act_frame, counts = activity_model.process_frame(frame_bgr.copy())

        # Model 2: crowd group detection
        crowd_frame = crowd_model.process_frame(frame_bgr.copy())

        act_rgb = cv2.cvtColor(act_frame, cv2.COLOR_BGR2RGB)
        crowd_rgb = cv2.cvtColor(crowd_frame, cv2.COLOR_BGR2RGB)

        placeholder_act.image(act_rgb, channels="RGB", use_container_width=True)
        placeholder_crowd.image(crowd_rgb, channels="RGB", use_container_width=True)

        # Optional keyboard quit when running locally
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()


# =================== SIDEBAR CONTROLS ===================

st.sidebar.header("Controls")

mode = st.sidebar.radio(
    "Select Source",
    ("Iriun Webcam", "Laptop Webcam", "Upload Video"),
)

start_clicked = st.sidebar.button("▶ Start")
stop_clicked = st.sidebar.button("■ Stop")

uploaded_file = None
if mode == "Upload Video":
    uploaded_file = st.sidebar.file_uploader(
        "Upload a video file",
        type=["mp4", "avi", "mkv"],
        key="uploader"
    )

if stop_clicked:
    st.session_state["run_iriun"] = False
    st.session_state["run_laptop"] = False
    st.session_state["run_upload"] = False

# =================== MAIN LOGIC PER MODE ===================

if mode == "Iriun Webcam":
    st.header("Iriun Webcam (Mobile Camera)")

    if start_clicked:
        st.session_state["run_iriun"] = True
        st.session_state["run_laptop"] = False
        st.session_state["run_upload"] = False

    if st.session_state["run_iriun"]:
        cap = open_iriun_camera(index=1)  # change index if needed
        run_stream("iriun", cap)

elif mode == "Laptop Webcam":
    st.header("Laptop Webcam")

    if start_clicked:
        st.session_state["run_laptop"] = True
        st.session_state["run_iriun"] = False
        st.session_state["run_upload"] = False

    if st.session_state["run_laptop"]:
        cap = open_laptop_camera(index=0)
        run_stream("laptop", cap)

elif mode == "Upload Video":
    st.header("Upload Video File")

    if start_clicked:
        if uploaded_file is None:
            st.error("Please upload a video first from sidebar.")
            st.session_state["run_upload"] = False
        else:
            st.session_state["run_upload"] = True
            st.session_state["run_iriun"] = False
            st.session_state["run_laptop"] = False

            path = save_uploaded_to_temp(uploaded_file)
            st.session_state["upload_path"] = path

    if st.session_state["run_upload"] and st.session_state["upload_path"] is not None:
        cap = open_video_file(st.session_state["upload_path"])
        run_stream("upload", cap)
