import cv2
import mediapipe as mp
import csv
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

face_mesh= mp_face_mesh.FaceMesh(max_num_faces=1,
min_detection_confidence=0.5,
min_tracking_confidence=0.5)


def face_point(image,drawing=False):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)
    faces=[]
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            face = []
            for id,lm in enumerate(faceLms.landmark):
                ih, iw, ic = image.shape
                x,y = int(lm.x*iw), int(lm.y*ih)
                face.append([id,x,y])     
            faces.append(face)
    image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image,faces

def mask_overlay(main_face,faces,mask_):
    mirror_point={
    234 :1,
    93 :2,
    132 :3,
    58 :4,
    172 :5,
    136 :6,
    150 :7,
    149 :8,
    176 :9,
    148 :10,
    152 :11,
    377 :12,
    400 :13,
    378 :14,
    379 :15,
    365 :16,
    397 :17,
    288 :18,
    361 :19,
    323 :20,
    454 :21,
    356 :22,
    389 :23,
    251 :24,
    284 :25,
    332 :26,
    297 :27,
    338 :28,
    10 :29,
    109 :30,
    67 :31,
    103 :32,
    54 :33,
    21 :34,
    162 :35,
    127 :36,
        }

    mask_annotation="./squid.csv"
    mask_points={}
    with open(mask_annotation) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for i, row in enumerate(csv_reader):
            mask_points[int(row[0])]=[float(row[1]), float(row[2])]
    src_pts = []
    for i in sorted(mask_points.keys()):
        try:
            src_pts.append(np.array(mask_points[i]))
        except ValueError:
            continue
    src_pts = np.array(src_pts, dtype="float32")
    face_points={}
    for i in faces[0]:
        for j in mirror_point.keys():
            if i[0] ==j:
                face_points[mirror_point[j]]=[float(i[1]),float(i[2])]
    dst_pts=[]
    for i in sorted(face_points.keys()):
        # print(i,face_points[i])
        try:
            dst_pts.append(np.array(face_points[i]))
        except ValueError:
            continue
    dst_pts = np.array(dst_pts, dtype="float32") 

    positions2=[]
    for i in face_points.keys():
        positions2.append([int(face_points[i][0]),int(face_points[i][1])])
    positions=[]
    for i in mask_points.keys():
        positions.append([int(mask_points[i][0]),int(mask_points[i][1])])
    
    
    building = main_face
    dp = mask_


    height, width = building.shape[:2]
    h1,w1 = dp.shape[:2]

    pts1=src_pts
    pts2= dst_pts
    # h, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC,5.0)
    h, mask = cv2.findHomography(pts1, pts2)


    height, width, channels = building.shape
    im1Reg = cv2.warpPerspective(dp, h, (width, height),
    None,
    cv2.INTER_LINEAR,
    cv2.BORDER_CONSTANT,)
   
    mask2 = np.zeros(building.shape, dtype=np.uint8)

    roi_corners2 = np.int32(positions2)

    channel_count2 = building.shape[2]  
    ignore_mask_color2 = (255,)*channel_count2

    cv2.fillConvexPoly(mask2, roi_corners2, ignore_mask_color2)

    mask2 = cv2.bitwise_not(mask2)
    masked_image2 = cv2.bitwise_and(building, mask2)

    final = cv2.bitwise_or(im1Reg, masked_image2)
    return final


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    mask_=cv2.imread("./squid.png")
    # mask_img = cv2.imread("./squid.png", cv2.IMREAD_UNCHANGED)
    # mask_img = mask_img.astype(np.float32)
    # mask_img = mask_img / 255.0
    # mask_=mask_img
    while True:
        success, image = cap.read()
        image=cv2.flip(image, 1)
        img,faces=face_point(image,drawing=False)
        if len(faces)>=1:
            image=mask_overlay(image,faces,mask_)
        cv2.imshow("Image", image)
        cv2.imwrite("Demo1.png",image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()
