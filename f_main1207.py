# f_main_modified.py

import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import platform
import sys
from pathlib import Path
import torch
import time
#回傳
import requests
#資料庫
import sqlite3
import pandas as pd

# 1. 連接到 SQLite 資料庫（如果不存在則創建）
db = sqlite3.connect('yydata_Tyy.db')  # 資料庫名稱
cursor = db.cursor()  # 創建游標

# --- (此處上方的大量 import 和設定保持不變) ---
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from for_yolov9.models.common import DetectMultiBackend
from for_yolov9.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from for_yolov9.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                                       increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from for_yolov9.utils.plots import Annotator, colors, save_one_box
from for_yolov9.utils.torch_utils import select_device, smart_inference_mode
from age_gender import age_gender_detect
from my_mediapipe import mediapipe_main
from algorithm_fordatabase_new import calculating
from face_opencv import face_run_test
from monodepth2 import monodepth_parse_args, monodepth_test_simple

# --- (此處上方的 send_total2 函式等保持不變) ---
def send_total2(s_total):
    raspberry_pi_url = "http://10.21.15.69:5002/update"
    data = str(s_total)
    try:
        response = requests.post(raspberry_pi_url, json=data)
        if response.status_code == 200:
            print(f"成功傳送數值至樹莓派")
        else:
            print(f"傳送至樹莓派失敗, 狀態碼: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"無法連接至樹莓派: {e}")
    return

c_time=time.time()

@smart_inference_mode()
def run(
        # ... (函式參數保持不變)
        weights= r"C:\Users\ben7u\OneDrive\Desktop\交接\program_origin\for_yolov9\exp39\weights\best.pt",
        source=ROOT / 'data/images',
        data=ROOT / 'data/coco.yaml',
        imgsz=(192, 640),
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
        device='',
        view_img=False,
        save_txt=False,
        save_conf=False,
        save_crop=False,
        nosave=False,
        classes=None,
        agnostic_nms=False,
        augment=False,
        visualize=False,
        update=False,
        project=ROOT / 'runs/detect',
        name='exp',
        exist_ok=False,
        line_thickness=3,
        hide_labels=False,
        hide_conf=False,
        half=False,
        dnn=False,
        vid_stride=1,
):

    c = 0
    XYWH = 0 
    reimagz = 0
    # countnum = 0  # <<< MODIFIED: 此變數已不需要，將其刪除
    savepp = './static' # 確保有一個名為 static 的資料夾來存放圖片
    
    # 從資料庫獲取目前最大的ID，以避免重啟時覆蓋
    try:
        cursor.execute("SELECT MAX(Book_ID) FROM fight WHERE Book_ID != 1000")
        max_id = cursor.fetchone()[0]
        ID = (max_id or 0) + 1
    except:
        ID = 1
    print(f"Starting with Book_ID: {ID}")

    global c_time
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)

    save_dir = Path(project) / name
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
    
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    bs = 1
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    vid_path, vid_writer = [None] * bs, [None] * bs
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        for i, det in enumerate(pred):
            s_yolo = 0
            seen += 1
            if webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
            s += '%gx%g ' % im.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy() if save_crop else im0
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                for *xyxy, conf, cls in reversed(det):
                    c_time=time.time()
                    if save_txt:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        X=int(xywh[0]*640); Y=int(xywh[1]*192); W=int(xywh[2]*640); H=int(xywh[3]*192)
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        if cls == 0: yolo_label = "people"; s_yolo = s_yolo + 1000
                        elif cls == 1: yolo_label = "wokon"; s_yolo = s_yolo + 8000
                        elif cls == 2: yolo_label = "knife"; s_yolo = s_yolo + 1600
                        elif cls == 3: yolo_label = "bat"; s_yolo = s_yolo + 1200
                        elif cls == 4: yolo_label = "gun"; s_yolo = s_yolo + 4000
                            
                    if save_img or save_crop or view_img:
                        c = int(cls)
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        
                        # <<< MODIFIED START >>>
                        # 關鍵修改：使用資料庫的 ID 來命名圖片，確保詳情頁能正確找到圖片
                        filename = f"{savepp}/{ID}.jpg"
                        # <<< MODIFIED END >>>
                        
                        cv2.imwrite(filename, im0)
                        
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                    
                    # --- 後續分析 (保持不變) ---
                    # --- (修改後的程式碼，增加了保護機制) ---
                    try:
                        distance = monodepth_test_simple(filename, X, Y, W, H)
                    except Exception:
                        distance = 0.0 # 發生錯誤時給予預設值

                    try:
                        gender, agess = age_gender_detect(filename)
                    except Exception:
                        gender, agess = "N/A", "N/A" # 發生錯誤時給予預設值

                    try:
                        all_pose, pose = mediapipe_main(filename)
                    except Exception:
                        all_pose, pose = "N/A", "N/A" # 發生錯誤時給予預設值

                    try:
                        emo = face_run_test(filename)
                    except Exception:
                        emo = "N/A" # 發生錯誤時給予預設值

                    s_total = calculating(gender, agess, emo, distance, pose)

                    s_total = s_total + s_yolo
                    print(f"Book_ID: {ID}, Total: {s_total}")

                    DD = float(distance)
                    cursor.execute("INSERT OR REPLACE INTO fight (Book_ID, Yolo, Distance,Face,Pose,Age,Gender,Total) VALUES (?,?,?,?,?,?,?,?)", (ID, yolo_label, DD,emo,all_pose,agess,gender,s_total))
                    db.commit()

                    # send_total2(s_total) # 如果需要，可以取消註解此行來發送數據到樹莓派
                    
                    ID = ID + 1

                    c_time2=time.time()
                    ttime = c_time2 - c_time
                    print(f'use time: {ttime:.4f}s')

                    # Line Notify 邏輯 (保持不變)
                    if s_total > 3275:
                        pass # 您的 Line Notify 程式碼可以放在這裡

            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if save_img:
                if dataset.mode == 'image': cv2.imwrite(save_path, im0)
                else:
                    if vid_path[i] != save_path:
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter): vid_writer[i].release()
                        if vid_cap: fps = vid_cap.get(cv2.CAP_PROP_FPS); w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else: fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # 檢查是否按下 'q' 來退出循環
        if 'q' in [cv2.waitKey(1) & 0xFF]:
            break
    
    db.close()
    print('Database connection closed.')
    LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
    t = tuple(x.t / seen * 1E3 for x in dt)
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)


# --- (此處下方的 yolo_parse_opt 和 main_yolo 函式保持不變) ---
def yolo_parse_opt(weights, source, img, device):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=weights, help='model path or triton URL')
    parser.add_argument('--source', type=str, default=source, help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[192, 640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default=device, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', default=True, help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/yy', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt


def main_yolo(opt):
    c, XYWH = run(**vars(opt))
    return c, XYWH


if __name__ == "__main__":
    weights_path = r"C:\Users\ben7u\OneDrive\Desktop\交接\program_origin\for_yolov9\exp29(1118all)\weights\best.pt"
    source = "rtsp://10.21.15.69:5000/stream"
    img = [640,192]
    device = 0
    
    opt = yolo_parse_opt(weights_path, source, img, device)
    opt.nosave = False 
    main_yolo(opt)