import cv2

def main():
    # Simplified GStreamer pipeline for IMX219 camera
    gst_pipeline = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=640, height=480, format=NV12 ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "appsink"
    )

    # Create a VideoCapture object with the GStreamer pipeline
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Failed to open camera.")
        print("GStreamer pipeline:", gst_pipeline)
        return

    # Create an OpenCV window
    cv2.namedWindow("Camera Stream", cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            # Capture a frame from the camera
            ret, frame = cap.read()

            if not ret:
                print("Failed to grab frame.")
                break

            # Display the frame in the OpenCV window
            cv2.imshow("Camera Stream", frame)

            # Check for exit key (ESC)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
