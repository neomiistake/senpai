import cv2
import math
import argparse
import time

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes

global gender
global age

def age_gender_detect(source):
    # source = 'rtsp://192.168.0.9:5000/stream'
    parser=argparse.ArgumentParser()
    # parser.add_argument('--image', type=str, default=picture)
    #parser.add_argument('--image', type=str, default='D:\fighting\runs\detect\exp22\labels\stream_65.jpg')
    parser.add_argument('--source', type=str, default=source, help='Input source: webcam or RTSP URL')

    args=parser.parse_args()
    
    faceProto="for_age_gender/opencv_face_detector.pbtxt"
    faceModel="for_age_gender/opencv_face_detector_uint8.pb"
    ageProto="for_age_gender/age_deploy.prototxt"
    ageModel="for_age_gender/age_net.caffemodel"
    genderProto="for_age_gender/gender_deploy.prototxt"
    genderModel="for_age_gender/gender_net.caffemodel"

    MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
    ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList=['Male','Female']

    faceNet=cv2.dnn.readNet(faceModel,faceProto)
    ageNet=cv2.dnn.readNet(ageModel,ageProto)
    genderNet=cv2.dnn.readNet(genderModel,genderProto)

    # video=cv2.VideoCapture(args.source) #串流
    # video=cv2.VideoCapture(args.image if args.image else 0) #圖片
    video=cv2.VideoCapture(args.source if args.source else 0)

    padding=20
    while cv2.waitKey(1)<0 :
        hasFrame,frame=video.read()
        if not hasFrame:
            print('age_gender : No frame received from the stream')
            # cv2.waitKey()
            break
        
        resultImg,faceBoxes=highlightFace(faceNet,frame)
        if not faceBoxes:
            # print(" age_gender_detect No face detected")
            gender = 'No'
            agess = 'No'
            
        for faceBox in faceBoxes:
            face=frame[max(0,faceBox[1]-padding):
                    min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                    :min(faceBox[2]+padding, frame.shape[1]-1)]

            blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds=genderNet.forward()
            gender=genderList[genderPreds[0].argmax()]
            # print(f'Gender: {gender}')

            ageNet.setInput(blob)
            agePreds=ageNet.forward()
            age=ageList[agePreds[0].argmax()]
            # print(f'Age: {age[1:-1]} years')
            agess  = age[1:-1]

            cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)

            # cv2.imshow("Detecting age and gender", resultImg)
        return gender, agess

    #tt+
    # video.release()
    # cv2.destroyAllWindows()
    

# source = r'D:\fighting2\38.jpg'
# gender, agess = age_gender_detect(source)
# c_time=time.time()
# img = r'D:\fighting\runs\detect\exp22\crops\people\stream68.jpg'
# gender, agess = age_gender_detect()
# print(agess, gender)
# time=time.time()-c_time
# print('use time:', time)