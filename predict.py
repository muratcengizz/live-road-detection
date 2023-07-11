from ultralytics import YOLO
import os 
import cv2

model = YOLO("best.pt")

path = os.chdir("C:/Users/murat/Documents/computer_vision3/live_road_detection/test/images")
files = os.listdir(path)

for image in files:
    img = cv2.imread(filename=image)
    
    results = model.predict(img)
    plotted = results[0].plot()
    

    cv2.imshow(winname="detection", mat=plotted)
    if cv2.waitKey(0) == ord("q"): continue
    
cv2.destroyAllWindows()