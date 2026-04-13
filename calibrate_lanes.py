import cv2

# --- CONFIGURATION ---
# Put the exact name of the video file you are testing here
VIDEO_PATH = 'uploads/your_video_name.mp4'

points = []


def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])

        # Draw a dot where you clicked
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

        # Connect the dots with a line
        if len(points) > 1:
            cv2.line(img, tuple(points[-2]),
                     tuple(points[-1]), (0, 255, 255), 2)

        cv2.imshow('Calibration Tool', img)

        if len(points) == 4:
            # Connect the last point to the first point to close the shape
            cv2.line(img, tuple(points[-1]),
                     tuple(points[0]), (0, 255, 255), 2)
            cv2.imshow('Calibration Tool', img)
            print("\n--- Copy this array into your app.py ---")
            print(f"np.array({points}, np.int32).reshape((-1, 1, 2))")
            print("----------------------------------------\n")
            points.clear()  # Reset for the next lane


# Read the first frame of the video
cap = cv2.VideoCapture(VIDEO_PATH)
ret, img = cap.read()
if not ret:
    print("Could not read video. Check the VIDEO_PATH.")
    exit()

cv2.imshow('Calibration Tool', img)
cv2.setMouseCallback('Calibration Tool', click_event)

print("Click 4 corners of a lane (Clockwise starting from top-left).")
print("Press 'q' or ESC to quit when done.")

cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()
