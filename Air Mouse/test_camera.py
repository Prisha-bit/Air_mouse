import cv2

cap = cv2.VideoCapture(0)  # Try 0, 1, or 2 if needed

while True:
    success, img = cap.read()
    if not success:
        print("Camera not detected. Try changing the index (0,1,2).")
        break

    cv2.imshow("Test", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
