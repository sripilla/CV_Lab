import cv2

# Step 1: Read the image
img = cv2.imread("sample.png")   # make sure sample.png is in CV_Lab

if img is None:
    print("❌ Could not read image. Check filename/path.")
else:
    # Step 2: Display the image
    cv2.imshow("Displayed Image", img)

    # Wait for a key press
    cv2.waitKey(0)

    # Close window
    cv2.destroyAllWindows()

    # Step 3: Save it
    cv2.imwrite("output.png", img)
    print("✅ Image displayed and saved as output.png")
