
#Package need to install 
#pip install ultralytics

from ultralytics import YOLO
import cv2

modelfile='runs/train/exp/weights/best.pt'
model = YOLO(modelfile)
print(model.names)

if False :
    image_path = 'test/images/10.jpg'
    results = model.predict(source=image_path)
    print("Bounding Boxes :",results[0].boxes.xyxy)
    print("Classes :",results[0].boxes.cls.tolist())
    for obj in results[0].boxes.cls.tolist() :
        print(model.names[obj])



def plot_rectangles(image, rectangles,cls):

    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Add rectangles to the plot
    for rect,c in zip(rectangles,cls):
        print(rect,c,model.names[c])
        
        if c==0:
            color=(0, 255, 0)
        elif c==1:
            color=(0, 255, 255)
            
        x1,y1,x2,y2= list(map(int,rect))

        #print(x1,y1,x2,y2)
        cv2.rectangle(image,(x1, y1), (x2, y2), color,2)
    
    return image
       

if False :
    # Call the function to plot the rectangles on the image
    # Load the image
    image = cv2.imread(image_path)
    frame_with_rects=plot_rectangles(image,results[0].boxes.xyxy.tolist(),results[0].boxes.cls.tolist())
    cv2.imshow('Video with Rectangles', frame_with_rects)
    while True:
        key = cv2.waitKey(90)
        if key == ord('q') or key == 27: # Esc
            print('break')
            break





video_path = 'kyrie.mp4'

video = cv2.VideoCapture(video_path)
success = True

while success:
  success,frame = video.read()
  if success == True:
    #cv2.imwrite(name,frame)
    results = model.predict(source=frame)
    frame_with_rects=plot_rectangles(frame,results[0].boxes.xyxy.tolist(),results[0].boxes.cls.tolist())
    cv2.imshow('Video with Rectangles', frame_with_rects)
  else:
    break
    
  
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break


    
    
# Release the VideoCapture and close all OpenCV windows
video.release()
cv2.destroyAllWindows()
