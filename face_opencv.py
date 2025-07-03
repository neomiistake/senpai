import os
import sys
import argparse
import numpy as np
import torch
from torchvision import transforms, datasets
import torch.utils.data as data
from for_DDAMFN.networks.DDAM import DDAMNet
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from PIL import Image, ImageFilter
import cv2
import time

# face_cascade_path = 'D:/DDAMFN++/haarcascade_frontalface_default.xml'


def parse_args(image):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--aff_path', type=str, default='test', help='AfectNet dataset path.')
    # parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    # parser.add_argument('--workers', default=8, type=int, help='Number of data loading workers.')
    parser.add_argument('--num_head', type=int, default=2, help='Number of attention heads.')
    parser.add_argument('--num_class', type=int, default=7, help='Number of classes.')
    parser.add_argument('--model_path', default='./for_DDAMFN/checkpoints_ver2.0/affecnet7_epoch19_acc0.671.pth')
    parser.add_argument('--image_path', type=str, default=image, help='Path of the image to predict.')
    return parser.parse_args()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j] * 100, fmt) + '%',
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Actual', fontsize=18)
    plt.xlabel('Predicted', fontsize=18)
    plt.tight_layout()

class7_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry']
class8_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry', 'Contempt']

def predict_single_image(image_path, model, device, num_class):
    data_transforms = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 加載圖片並應用轉換
    image = Image.open(image_path).convert('RGB')
    
    
    #轉成opencv
    opencv_img=np.array(image)
    bgr=cv2.cvtColor(opencv_img, cv2.COLOR_RGB2BGR)

    

    #把臉的部分給截下來
    face_cas=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces=face_cas.detectMultiScale(bgr, 1.1, 3)

    if len(faces) > 0:  # 或者使用 faces.any()
        for (x, y, w, h) in faces:
            ROI = bgr[y:y+h, x:x+w]
    # 轉成PIL
        rgb = cv2.cvtColor(ROI, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
    else:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)



    #predict
    transformed_image = data_transforms(pil_img).unsqueeze(0)  # 添加 batch 維度
    transformed_image = transformed_image.to(device)

    # 預測
    model.eval()
    with torch.no_grad():
        output, _, _ = model(transformed_image)
        _, prediction = torch.max(output, 1)

    return prediction.item(), transformed_image.squeeze(0).cpu()  # 回傳預測結果和轉換後的圖片張量

def capture_image_from_camera():
    # Capture a single frame from the camera
    cap = cv2.VideoCapture(0)  # 0 is usually the default camera
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Unable to capture image from camera.")
        return None
    
    # Convert captured frame (OpenCV format) to PIL format
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def face_run_test(image):

  

    args = parse_args(image)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = DDAMNet(num_class=args.num_class, num_head=args.num_head, pretrained=False)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # 如果提供了圖片路徑，則進行單張圖片預測
    if args.image_path:
        prediction, transformed_image = predict_single_image(args.image_path, model, device, args.num_class)
        
        if args.image_path == 'camera':
            image = capture_image_from_camera()
            if image is None:
                return
            # Optionally save the captured image
            image.save('captured_image.jpg')
            image_path = 'captured_image.jpg'
        else:
            image_path = args.image_path

        prediction, transformed_image = predict_single_image(image_path, model, device, args.num_class)

        # if args.num_class == 7:
            # print(f"Predicted class: {class7_names[prediction]}")
            
            # Display the image
            # np_image = transformed_image.permute(1, 2, 0).numpy()  # Convert tensor to numpy format
            # np_image = np.clip(np_image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)  # De-normalize
            # plt.imshow(np_image)
            # plt.title(f"Predicted class: {class7_names[prediction]}")
            # plt.show()

        # elif args.num_class == 8:
            # print(f"Predicted class: {class8_names[prediction]}")
            
            # Display the image
            # np_image = transformed_image.permute(1, 2, 0).numpy()  # Convert tensor to numpy format
            # np_image = np.clip(np_image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)  # De-normalize
            # plt.imshow(np_image)
            # plt.title(f"Predicted class: {class8_names[prediction]}")
            # plt.show()
    return class7_names[prediction]
    
    
#if __name__ == "__main__":
 #   run_test()
