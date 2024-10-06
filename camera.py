import cv2
import base64
import numpy as np
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

# Server URL (update if necessary)
SERVER_URL = 'http://129.213.16.27:5000/process_image'

# Number of concurrent requests
MAX_CONCURRENT_REQUESTS = 1

async def send_frame(session, frame):
    # Encode frame to base64
    _, buffer = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    try:
        async with session.post(SERVER_URL, json={'image': img_base64}) as response:
            if response.status == 200:
                response_data = await response.json()
                processed_img_data = base64.b64decode(response_data['image'])
                processed_img_arr = np.frombuffer(processed_img_data, np.uint8)
                return cv2.imdecode(processed_img_arr, cv2.IMREAD_COLOR)
            else:
                print(f"Error: Server returned status code {response.status}")
                return None
    except Exception as e:
        print(f"Error sending request: {e}")
        return None

async def process_frames(frames):
    async with aiohttp.ClientSession() as session:
        tasks = [send_frame(session, frame) for frame in frames]
        return await asyncio.gather(*tasks)

async def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {frame_width}x{frame_height}")

    with ThreadPoolExecutor(max_workers=1) as executor:
        while True:
            frames = []
            for _ in range(MAX_CONCURRENT_REQUESTS):
                print("sending request")
                ret, frame = await asyncio.get_event_loop().run_in_executor(executor, cap.read)
                if not ret or frame is None or frame.size == 0:
                    print("Failed to capture frame")
                    continue
                frames.append(frame)

            if not frames:
                continue

            processed_frames = await process_frames(frames)

            for processed_frame in processed_frames:
                if processed_frame is not None:
                    print("showing request")
                    cv2.imshow("Processed Segmentation", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())