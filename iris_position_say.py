import cv2 as cv
import numpy as np 
import mediapipe as mp
import math 

mp_face_mesh = mp.solutions.face_mesh

right_iris = [474,475,476,477]
left_iris = [469,470,471,472]
      
L_H_LEFT = [33]  
L_H_RIGHT = [133]  
R_H_LEFT = [362]  
R_H_RIGHT = [263] 

def euclidean_distance(point1,point2):
        x1,y1 = point1.ravel()
        x2,y2 = point2.ravel()
        distance = math.sqrt((x2-x1)**2+(y2-y1))
        return distance

def iris_position(iris_center,right_point,left_point):
       centre_to_right_dist =  euclidean_distance(iris_center,right_point)
       total_distance =  euclidean_distance(right_point,left_point)
       ratio= centre_to_right_dist/total_distance
       iris_position = ""
       if ratio <=0.42:
              iris_position = "right"
       elif ratio>0.42 and ratio<= 0.57:
              iris_position = "center"
       else:
              iris_position = "left"
       return iris_position,ratio


cap = cv.VideoCapture(0)
with mp_face_mesh.FaceMesh(
       
    max_num_faces = 1,
    refine_landmarks = True,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5

) as face_mesh:
    while True:
            ret,frame = cap.read()
            if not ret:
                    break
            
            frame = cv.flip(frame,1)
            rgb_frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
            img_h,img_W = frame.shape[:2]
            resolt = face_mesh.process(rgb_frame)
            if resolt.multi_face_landmarks:
                   mesh_point =np.array([np.multiply([p.x,p.y],[img_W,img_h]).astype(int)for p in resolt.multi_face_landmarks[0].landmark])
                   (l_cx, l_cy),l_radious =  cv.minEnclosingCircle(mesh_point[left_iris])
                   (r_cx, r_cy),r_radious =  cv.minEnclosingCircle(mesh_point[right_iris])

                   centre_left = np.array([l_cx,l_cy],dtype=np.int32)
                   centre_rigth = np.array([r_cx,r_cy],dtype=np.int32)

                   cv.circle(frame,centre_left,int(l_radious),(0,0,255),1,cv.LINE_AA)
                   cv.circle(frame,centre_rigth,int(r_radious),(0,0,255),1,cv.LINE_AA)

                   iris_pos,ratio = iris_position(centre_rigth,mesh_point[R_H_RIGHT],mesh_point[R_H_LEFT][0])
                   cv.putText(frame,f"iris position:{iris_pos} {ratio:.2f}",(30,30),cv.FONT_HERSHEY_PLAIN,1.2,(0,255,0),1,cv.LINE_AA)
                   
            cv.imshow('img',frame)            
            key = cv.waitKey(1)
            if key == ord('q'):
                    break
cap.release()
cv.destroyAllWindows()

                   