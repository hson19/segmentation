import cv2
import requests
import base64
import numpy as np

cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
# Server URL
SERVER_URL = 'http://129.213.95.239:5000/process_image'
print("test")
# Open webcam
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Get the actual frame size
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera resolution: {frame_width}x{frame_height}")

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break
    # Check if the frame is empty
    if frame is None or frame.size == 0:
        print("Captured frame is empty")
        continue

    # Display the original frame
    # cv2.imshow("Original", frame)
    frame_count = 0
    while frame_count < 10:  # Capture 10 frames
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Check if the frame is empty
        if frame is None or frame.size == 0:
            print("Captured frame is empty")
            continue

        # Save the frame as an image file
        cv2.imwrite(f"frames/frame_{frame_count}.jpg", frame)
        print(f"Saved frame_{frame_count}.jpg")

        frame_count += 1

    # Encode frame to base64
    _, buffer = cv2.imencode('.jpg', frame)

    img_base64 = base64.b64encode(buffer).decode('utf-8')


    # Send frame to server
    response = requests.post(SERVER_URL, json={'image': img_base64})
    print("Response ???",response)
     
    if response.status_code == 200:
        print("statut code 200")
        # Decode the processed image
        processed_img_data = base64.b64decode(response.json()['image'])
        processed_img_arr = np.frombuffer(processed_img_data, np.uint8)
        processed_img = cv2.imdecode(processed_img_arr, cv2.IMREAD_COLOR)

        # Display the processed image
        # cv2.imshow("Processed Segmentation", processed_img)
        cv2.imwrite(f"frames/frame_{frame_count}.jpg", processed_img)
        print(f"Saved frame_{frame_count}.jpg")
    else:
        print(f"Error: Server returned status code {response.status_code}")

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()