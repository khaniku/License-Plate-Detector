import matplotlib.pyplot as plt
import imutils
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import shutil
from skimage import measure
from skimage.measure import regionprops
from skimage.io import imread
from skimage.filters import threshold_otsu

filename = './video.mp4'

if os.path.exists('output'):
    shutil.rmtree('output')

os.makedirs('output')

cap = cv2.VideoCapture(filename)
count = 0
while cap.isOpened():
    ret,frame = cap.read()
    if ret == True:
        cv2.imshow('window-name',frame)
        cv2.imwrite("./output/frame%d.jpg" % count, frame)
        count = count + 1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()

car_image = imread("./output/frame%d.jpg"%(count-1), as_gray=True)
car_image = imutils.rotate(car_image, 90)
print(car_image.shape)

gray_car_image = car_image * 255
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(gray_car_image, cmap="gray")
threshold_value = threshold_otsu(gray_car_image)
binary_car_image = gray_car_image > threshold_value
ax2.imshow(binary_car_image, cmap="gray")
plt.show()


label_image = measure.label(binary_car_image)

plate_dimensions = (0.08*label_image.shape[0], 0.2*label_image.shape[0], 0.15*label_image.shape[1], 0.42*label_image.shape[1])
min_height, max_height, min_width, max_width = plate_dimensions
plate_objects_cordinates = []
plate_like_objects = []

fig, (ax1) = plt.subplots(1)
ax1.imshow(gray_car_image, cmap="gray")

for region in regionprops(label_image):
    if region.area < 50:
        continue

    min_row, min_col, max_row, max_col = region.bbox

    region_height = max_row - min_row
    region_width = max_col - min_col

    if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:
        plate_like_objects.append(binary_car_image[min_row:max_row,
                                  min_col:max_col])
        plate_objects_cordinates.append((min_row, min_col,
                                         max_row, max_col))
        rectBorder = patches.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row, edgecolor="red",
                                       linewidth=2, fill=False)
        ax1.add_patch(rectBorder)

plt.show()
