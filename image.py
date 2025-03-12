import cv2
import matplotlib.pyplot as plt
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
model = cv2.dnn_DetectionModel(frozen_model,config_file)
model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)
labels = []
file_name = 'label.txt'
with open(file_name,'rt') as file:
    labels = file.read().rstrip('\n').split('\n')

#print("Total number of labels: ",len(labels))
#print("The labels are: \n",labels)
img1 = cv2.imread('test1.jpg')
plt.imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)) # converted the image from bgr to rgb
#ClassIndex, confidence, bbox = model.detect(img1,confThreshold=0.7)
#print(ClassIndex)
#font_scale = 3
#font = cv2.FONT_HERSHEY_PLAIN
#for ind, conf, box, in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
#    cv2.rectangle(img1,box,(225,0,0),2)
#    cv2.putText(img1,labels[ind-1],(box[0]+10,box[1]+40),font,fontScale=font_scale,color=(0,225,0),thickness=3)
#plt.imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
#img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
#cv2.imwrite('./img/result1.jpg',img1)
