




def calculating(gender,agess, emo, distance, pose):
    
    #gender
    #'Male' 1000, 'Female' 900
    #height
    # M>172: 100, M<172: 50, F>160: 100, F<160: 50
    if gender == 'Male':
        s_gender = 6
    elif gender == 'Female':
        s_gender = 5
    else:
        s_gender = 0


    #age
    #'(0-2)' 10, '(4-6)' 50, '(8-12)' 150, '(15-20)' 200, '(25-32)' 250, '(38-43)' 250, '(48-53)' 200, '(60-100)' 150
    if agess == '0-2':
        s_age = 0
    elif agess == '4-6':
        s_age = 1
    elif agess == '8-12':
        s_age = 3
    elif agess == '15-20':
        s_age = 7
    elif agess == '25-32':
        s_age = 6
    elif agess == '38-43':
        s_age = 5
    elif agess == '48-53':
        s_age = 4
    elif agess == '60-100':
        s_age = 3
    else:
        s_age = 0


    #emo
    #angry:350, disgust:250, fear:200, happy:50, sad:50, neutral:175, surprise:100
    if emo == 'Angry':
        s_emo = 6
    elif emo == 'Disgust':
        s_emo = 4
    elif emo == 'Fear':
        s_emo = 3
    elif emo == 'Happy':
        s_emo = 1
    elif emo == 'Sad':
        s_emo = 2
    elif emo == 'Neutral':
        s_emo = 3
    elif emo == 'Surprise':
        s_emo = 2
    else :
        s_emo = 0

    #distance
    #<1m: 300,  1~2m: 250, 2~3m: 100, >3m: 50
    if distance <= 1:
        s_distance = 6
    elif distance >= 1 and distance <= 2 :
        s_distance = 4
    elif distance >= 2 and distance <= 3 :
        s_distance = 2
    else :
        s_distance = 1

    #pose
    # if pose == 2:
    #     s_pose = 300
    #     # print("結論:危險姿勢")
    # elif pose == 1:
    #     s_pose = 150
    #     # print("結論:有點危險")
    # else :
    #     s_pose = 0
    #     # print("結論:安全")

    #pose
    # if pose == 2:
    #     s_pose = 300
    #     # print("結論:危險姿勢")
    # elif pose == 1:
    #     s_pose = 150
    #     # print("結論:有點危險")
    # else :
    #     s_pose = 0
    #     # print("結論:安全")
    s_pose=pose
    # if pose ==500:
    #     s_pose=500
    #     print("危險姿勢1")
    # elif pose==400:
    #     s_pose=400
    #     print("危險姿勢2")
    # elif pose==300:
    #     s_pose=300
    #     print("非危險姿勢1")
    # elif pose==200:
    #     s_pose=200
    #     print("非危險姿勢2")
    # elif pose==1000:
    #     s_pose=1000
    #     print("龜派氣功")
    # else:
    #     pose = 0
    # s_total=pose


  
    s_total = 50*s_gender + 50*s_age + 75*s_emo + 150*s_distance + 100*s_pose
    # s_total = 50*s_gender + 50*s_age + 75*s_emo + 150*s_distance + 100*s_pose


    


    #s_total = s_gender + s_age + s_emo + s_distance 

    return s_total