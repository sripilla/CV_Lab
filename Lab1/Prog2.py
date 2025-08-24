import cv2

# Open the video file
cap = cv2.VideoCapture("sample.mp4")   # make sure sample.mp4 is in the same folder

if not cap.isOpened():
    print("❌ Error: Could not open video file.")
    exit()

while True:
    # Read frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("✅ Video playback finished.")
        break
    
    # Display the frame
    cv2.imshow("Video Playback", frame)

    # Press 'q' to exit early
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# write a simple program to read and display a video file