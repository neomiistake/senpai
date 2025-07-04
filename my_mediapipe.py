import cv2
import mediapipe as mp
import numpy as np
import random as rd
import math

# 根據兩點的座標，計算角度  ((t+)應該是四點，兩向量)
def vector_2d_angle(v1, v2):
    v1_x = v1[0]
    v1_y = v1[1]
    v2_x = v2[0]
    v2_y = v2[1]
    try:
        angle_= math.degrees(math.acos((v1_x*v2_x+v1_y*v2_y)/ ( ((v1_x**2+v1_y**2)**0.5) * ((v2_x**2+v2_y**2)**0.5) )))
    except:
        angle_ = 180
    return angle_

# 根據傳入的 21 個節點座標，得到該手指的角度
def hand_angle(hand_):
    global hand_len
    hand_len=int(int(hand_[8][0])-int(hand_[4][0]))
    angle_list = []
    # thumb 大拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[2][0])),(int(hand_[0][1])-int(hand_[2][1]))),
        ((int(hand_[3][0])- int(hand_[4][0])),(int(hand_[3][1])- int(hand_[4][1])))
        )
    angle_list.append(angle_)
    # index 食指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])-int(hand_[6][0])),(int(hand_[0][1])- int(hand_[6][1]))),
        ((int(hand_[7][0])- int(hand_[8][0])),(int(hand_[7][1])- int(hand_[8][1])))
        )
    angle_list.append(angle_)
    # middle 中指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[10][0])),(int(hand_[0][1])- int(hand_[10][1]))),
        ((int(hand_[11][0])- int(hand_[12][0])),(int(hand_[11][1])- int(hand_[12][1])))
        )
    angle_list.append(angle_)
    # ring 無名指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[14][0])),(int(hand_[0][1])- int(hand_[14][1]))),
        ((int(hand_[15][0])- int(hand_[16][0])),(int(hand_[15][1])- int(hand_[16][1])))
        )
    angle_list.append(angle_)
    # pink 小拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[18][0])),(int(hand_[0][1])- int(hand_[18][1]))),
        ((int(hand_[19][0])- int(hand_[20][0])),(int(hand_[19][1])- int(hand_[20][1])))
        )
    angle_list.append(angle_)


    return angle_list

# 根據手指角度的串列內容，返回對應的手勢名稱
def hand_pos(finger_angle, hand_, depth_):
    f1 = finger_angle[0]   # 大拇指角度
    f2 = finger_angle[1]   # 食指角度
    f3 = finger_angle[2]   # 中指角度
    f4 = finger_angle[3]   # 無名指角度
    f5 = finger_angle[4]   # 小拇指角度

    # 小於 50 表示手指伸直，大於等於 50 表示手指捲縮
    if f1>=50 and f2>=50 and f3<50 and f4>=50 and f5>=50:
        return 'no!!!'
    elif f1<50 and f2<50 and f3>=50 and f4>=50 and f5<50:
        return 'ROCK!'
    elif f1>=50 and f2>=50 and f3>=50 and f4>=50 and f5>=50:
        return '0'
    elif f1>=50 and f2>=50 and f3>=50 and f4>=50 and f5<50:
        return 'pink'
    elif f1>=50 and f2<50 and f3>=50 and f4>=50 and f5>=50:
        return '1'
    elif f1>=50 and f2<50 and f3<50 and f4>=50 and f5>=50:
        return '2'
    elif f1>=50 and f2>=50 and f3<50 and f4<50 and f5<50:
        return 'ok'
    elif f1<50 and f2>=50 and f3<50 and f4<50 and f5<50:
        return 'ok'
    elif f1>=50 and f2<50 and f3<50 and f4<50 and f5>50:
        return '3'
    elif f1>=50 and f2<50 and f3<50 and f4<50 and f5<50:
        return '4'
    elif f1<50 and f2<50 and f3<50 and f4<50 and f5<50:
        return '5'
    elif f1<50 and f2>=50 and f3>=50 and f4>=50 and f5<50:
        return '6'
    # elif f1<50 and f2>=70 and f3>=50 and f4>=50 and f5>=50:
    #     return 'good'
    elif f1<50 and f3>=50 and f4>=50 and f5>=50:
        #f1<50 and f2<70 and f3>=50 and f4>=50 and f5>=50
        xx = int(hand_[4][0])-int(hand_[7][0])
        yy = int(hand_[4][1])-int(hand_[7][1])
        distance47 = pointTOpoint(xx, yy)
        xx = int(hand_[7][0])-int(hand_[8][0])
        yy = int(hand_[7][1])-int(hand_[8][1])
        distance78 = pointTOpoint(xx, yy)
        # print('d47: ' , distance47)
        # print('d78: ' , distance78)
        
        if f2>=60:
            if (distance47 - distance78)<=20:
                return 'love'
            else:
                return 'good'
        else :
            return '7'

    elif  f1<50 and f2<50 and f3<50 and f4>=50 and f5>=50:
        return '8'
    elif f1<50 and f2<50 and f3<50 and f4<50 and f5>=50:
        return '9'
    else:
        return ''

#(t+)計算點和點之間的距離
def pointTOpoint (xx, yy):
    distance_ = (xx**2 + yy**2)**0.5
    return distance_

def cal_angle(a,b,c):
    a=np.array(a)
    b=np.array(b)
    c=np.array(c)

    radians=np.arctan2(c[1]-b[1],c[0]-b[0])-np.arctan2(a[1]-b[1],a[0]-b[0])
    angle=np.abs(radians*180.0/np.pi)

    if angle>180.0:
        angle=360-angle
    
    return angle

def cal_distance(a,b):
    a=np.array(a)
    b=np.array(b)
    
    return np.sqrt(np.sum((a-b)**2))




def mediapipe_main(img_path):

    all_stage = None

    # 讀取輸入影像
    img = cv2.imread(img_path)
    if img is None:
        print("Cannot read the image. Please check the file path.")
        return
    
    w, h = 520, 300 #影像尺寸
    img = cv2.resize(img, (w, h))  # 縮小尺寸，加快演算速度
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 將 BGR 轉換成 RGB


    mp_drawing = mp.solutions.drawing_utils          # mediapipe 繪圖方法
    mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
    mp_pose = mp.solutions.pose                      # mediapipe 姿勢偵測
    mp_hands = mp.solutions.hands


    stage=None
    stage1=None
    stage2=None
    hand_stage=None

    stage_s=0
    stage1_s=0
    hand_stage_s=0

    
    r=rd.randint(0,255)
    g=rd.randint(0,255)
    b=rd.randint(0,255)

    counter = 0

    # fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    # out=cv2.VideoWriter('F:\mediapipe\output.mp4',fourcc,20.0,(w,h))
    

    # Process pose detection
    with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.5) as pose:
        pose_results = pose.process(img_rgb)

    # Process hand detection
    with mp_hands.Hands(model_complexity=1, min_detection_confidence=0.3, min_tracking_confidence=0.5) as hands:
        hand_results = hands.process(img_rgb)

    # Create a blank canvas for drawing
    canvas = np.zeros(img.shape, dtype='uint8')


    if pose_results.pose_landmarks:

        # Extract landmarks
        landmarks = pose_results.pose_landmarks.landmark
        
        #取得座標
        left_eye = [landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].y]
        left_hip= [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        left_index = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y]
        left_thumb = [landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].x,landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_fake_d = [left_heel[0]-50,left_heel[1]]

        right_eye = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
        right_index = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y] 
        right_fake_d = [right_heel[0]-50,right_heel[1]]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        right_thumb = [landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        
        #計算有沒有跌倒的距離
        #left side
        left_distance=cal_distance(left_eye,left_heel)
        left_floor=[(abs(left_fake_d[0]-left_heel[0])),left_fake_d[1]]
        #right side
        right_distance=cal_distance(right_eye,right_heel)
        right_floor=[(abs(right_fake_d[0]-right_heel[0])),right_fake_d[1]]

        #計算有沒有跌倒的角度
        left_angle=cal_angle(left_eye,left_heel,left_fake_d)
        right_angle=cal_angle(right_eye,right_heel,right_fake_d)
        #計算蹲坐站走的角度
        angle3=cal_angle(left_hip,left_knee,left_ankle)
        angle4=cal_angle(right_hip,right_knee,right_ankle)
        
        #計算蹲坐站走的距離
        distance1=cal_distance(right_hip,right_index)
        distance2=cal_distance(left_hip,left_index)
        distance3=cal_distance(left_knee,left_index)
        distance4=cal_distance(right_knee,right_index)
        dis_ankle = cal_distance(left_ankle, right_ankle)
        dis_knee = cal_distance(left_knee, right_knee)


        if angle3>170 and angle4>170:
            stage="stand"
            stage_s = 1
        elif angle3>170 and angle4<170 or angle3<170 and angle4>170:
            stage="walk"
            stage_s = 3
        elif dis_ankle > dis_knee:
            stage = "feet spread"
            stage_s = 2
        else:
            stage_s = 0
        

        #判斷蹲坐站走和跌倒
        # if left_angle<=60 or right_angle<=60:
        #     stage='fallen'
        # else:
        #     if angle3>170 and angle4>170:
        #         stage="stand"
        #     elif angle3>170 and angle4<170 or angle3<170 and angle4>170:
        #         stage="walk"
        #     else:
        #         if distance3>distance2 and distance4>distance1:
        #             stage="sit"
        #         else:
        #             stage="squat"
            #stage='safe'
        


        #判斷有沒有舉手(以wrist和elbow為基準)
        if (left_wrist and left_elbow) or (right_wrist and right_elbow):
            if (left_wrist[1]<left_elbow[1]) or (right_wrist[1]<right_elbow[1]):
                if left_wrist[1]<left_elbow[1]:
                    counter+=1
                if right_wrist[1]<right_elbow[1]:
                    counter+=1
                if counter==1:
                    stage1="Raise one hand"
                    stage1_s = 3
                elif counter==2:
                    stage1="Raise two hand"
                    stage1_s = 3
        else:
            stage1=" "
            stage1_s = 1
        
        #判斷有沒有揮手
        # distance5=cal_distance(previous_left_wrist,left_wrist)*1000
        # distance6=cal_distance(previous_right_wrist,right_wrist)*1000
        # distance7=cal_distance(left_eye,left_thumb)*1000
        # distance8=cal_distance(right_eye,right_thumb)*1000
        # print(previous_left_thumb,'\t',left_thumb)
        # print(distance5,'\t',distance6)
        # if 5<=(distance5 or distance6)<=30:
        #     stage2="wave hand"
        # else:
        #     stage2=" "



        # 根據姿勢偵測結果，標記身體節點和骨架
        mp_drawing.draw_landmarks(
            img,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        

    #手掌偵測
    if hand_results.multi_hand_landmarks:
        left_wrist = None
        right_wrist = None
        left_mid = None
        right_mid = None

        for hand_land_landmark, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            label = handedness.classification[0].label

            # Detect hand wrist position
            for idx, i in enumerate(hand_land_landmark.landmark):
                x = i.x * img.shape[1]
                y = i.y * img.shape[0]

                # if idx == 0:  # Wrist landmark
                #     if label == 'Left':
                #         left_wrist = (int(x), int(y))
                #         # cv2.circle(img, left_wrist, 10, (255, 0, 0), -1)
                #         # cv2.putText(img, "", (left_wrist[0], left_wrist[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                #     elif label == 'Right':
                #         right_wrist = (int(x), int(y))
                #         # cv2.circle(img, right_wrist, 10, (0, 255, 0), -1)
                #         # cv2.putText(img, "", (right_wrist[0], right_wrist[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                
                if label == 'Left':
                    if idx == 0:  # Wrist landmark
                        left_wrist = (int(x), int(y))
                    if idx == 12:
                        left_mid = (int(x), int(y))

                elif label == 'Right':
                    if idx == 0:
                        right_wrist = (int(x), int(y))
                    if idx == 12:
                        right_mid = (int(x), int(y))
 

            finger_points = []                   # 記錄手指節點座標的串列
            finger_depths = []
            if finger_points:
                finger_angle = hand_angle(finger_points) # 計算手指角度，回傳長度為 5 的串列
                # print(finger_angle)                     # 印出角度 ( 有需要就開啟註解 )
                hand_stage = hand_pos(finger_angle, finger_points, finger_depths)            # 取得手勢所回傳的內容
                # cv2.putText(img, text, (30,120), cv2.FONT_HERSHEY_SIMPLEX, 5, (r,g,b), 10, cv2.LINE_AA) # 印出文字

            if left_wrist and right_wrist:
                distance = cal_distance(left_wrist,right_wrist)
                if distance < 50:
                    
                    # cv2.line(img, left_wrist, right_wrist, (0, 0, 255), 3)
                    # cv2.putText(img, "Wrists Touch!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # hand_stage = "Touch"
                    # print("Touch")
                    if left_mid and right_mid:
                        # print(left_mid, right_mid)
                        d_mid_y = abs(left_mid[1] - right_mid[1])
                        # d_mid_y = pointTOpoint(left_mid[1],right_mid[1])
                        # print(d_mid_y)

                        hand_stage = "Touching"
                        # print("Touching")
                        hand_stage_s = 9
                        # if d_mid_y <= 50:
                        #     hand_stage = "Touching"
                        #     print("Touching")
                        #     hand_stage_s = 9
                    # dan_num5=1
                    # if stage_right == "Right Arm Straightening" and stage == "Left Arm Straightening":
                    #     cv2.putText(img, "attack!", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

            if hand_stage == None:
                hand_stage_s = 0



        # for hand_landmarks in hand_results.multi_hand_landmarks:
        #     finger_points = []                   # 記錄手指節點座標的串列
        #     finger_depths = []
        #     for i in hand_landmarks.landmark:
        #         # 將 21 個節點換算成座標，記錄到 finger_points
        #         x = i.x*w
        #         y = i.y*h
        #         finger_points.append((x,y))
        #         z = i.z*1000
        #         finger_depths.append(z)
        #     if finger_points:
        #         finger_angle = hand_angle(finger_points) # 計算手指角度，回傳長度為 5 的串列
        #         # print(finger_angle)                     # 印出角度 ( 有需要就開啟註解 )
        #         hand_stage = hand_pos(finger_angle, finger_points, finger_depths)            # 取得手勢所回傳的內容
        #         # cv2.putText(img, text, (30,120), cv2.FONT_HERSHEY_SIMPLEX, 5, (r,g,b), 10, cv2.LINE_AA) # 印出文字

            # if left_wrist and right_wrist:
            #     distance_wrists = cal_distance(left_wrist, right_wrist)
            #     if distance_wrists < 50:
            #         cv2.line(img, left_wrist, right_wrist, (0, 0, 255), 3)
            #         cv2.putText(canvas, "Wrists Touching!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #         # print("Touching")
            

            # 將節點和骨架繪製到影像中(t+)
            mp_drawing.draw_landmarks(
                img,
                hand_land_landmark,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
                


    cv2.putText(canvas,stage,(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(r,g,b),2,cv2.LINE_AA)
    cv2.putText(canvas,stage1,(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(r,g,b),2,cv2.LINE_AA)
    # cv2.putText(canvas,stage2,(10,70),cv2.FONT_HERSHEY_SIMPLEX,1,(r,g,b),2,cv2.LINE_AA)
    cv2.putText(canvas,hand_stage,(10,90),cv2.FONT_HERSHEY_SIMPLEX,1,(r,g,b),2,cv2.LINE_AA)
    print(stage, stage1, hand_stage)

    output=cv2.addWeighted(img,1,canvas,1,0)
   
        # out.write(output)
        # if cv2.waitKey(5) == ord('q'):
        #     break     # 按下 q 鍵停止
        
        # #儲存之前的座標
        # try:
        #     # previous_left_thumb=left_thumb
        #     # previous_right_thumb=right_thumb
        #     previous_left_wrist=left_wrist
        #     previous_right_wrist=right_wrist
        # except:
        #     pass

        # out.release()
    # cv2.imshow('output', output)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    s_pose = stage_s + stage1_s + hand_stage_s
    #print(s_pose)

    # if stage != None:
    #     all_stage = all_stage + stage
    # if stage1 != None:
    #     all_stage = all_stage + stage1
    # if hand_stage != None:
    #     all_stage = all_stage + hand_stage
    all_stage = str(stage)+str(stage1)+str(hand_stage)
    #print(all_stage)
    #print(type(all_stage))

    return all_stage,s_pose




# try:
#     cap = cv2.VideoCapture(0)
#     print('use cam0')
# except:
#     cap = cv2.VideoCapture(1)
#     print('use cam1')
# img_path = cap

# img_path = r'D:\fighting\runs\yy\23.jpg'
# mediapipe_main(img_path)