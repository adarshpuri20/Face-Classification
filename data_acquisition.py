import cv2
import numpy as np
import os
import glob

# for data acquisition

os.getcwd()
##os.chdir(r'D:\computer vision\project01')


labels = glob.glob('.\FDDB-folds\*.txt')



l_contents = []                 # list of lines of textfile

for j in range(len((labels))):
    with open(labels[j], 'r') as f:
        f_contents = f.readlines()
        for a in f_contents:
            a = a.rstrip("\n")
            l_contents.append(a)


tc = 0
ix = 0
for i in l_contents:
    try:
        i = int(i)
        if i==1:
         #path
            path_of_image = l_contents[ix-1]
            ##          print(l_contents[ix-1])

            #ellipse
            ##            print(l_contents[ix+1])
            temp = l_contents[ix+1].split(" ")
            maj_ax = round(float(temp[0]))
            min_ax = round(float(temp[1]))
            angle = round(float(temp[2]))
            cent_x = round(float(temp[3]))
            cent_y = round(float(temp[4]))


            img = cv2.imread((path_of_image + '.jpg'))

            ##            image = cv2.ellipse(img, (cent_x,cent_y), (min_ax,maj_ax), 0, 0, 360, (0,0,255), -1)

            (w,h) = img.shape[:2]
            w = 2*w
            h = 2*h
            M = cv2.getRotationMatrix2D((cent_x,cent_y), angle, 1)
            rotated = cv2.warpAffine(img, M, (w, h))

            crop_img = rotated[cent_y - maj_ax:cent_y + maj_ax, cent_x - min_ax:cent_x + min_ax]

            resized = cv2.resize(crop_img, (20,20), interpolation = cv2.INTER_AREA)

            img_nonf = img[0:20,0:20]

##            cv2.imwrite(f'.\data_set\face\image{tc}.jpg', resized)
            cv2.imwrite(f'.\data_set_f\{tc}.jpg', resized)
            

         
            cv2.imwrite(f'.\data_set_nf\{tc}.jpg', img_nonf)
            
            tc = tc + 1


    except Exception:
        pass
    ix = ix + 1