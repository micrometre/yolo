# Importing the EasyOCR library
import easyocr
import cv2
import re
import matplotlib.pyplot as plt

# Loading the OCR reader
reader = easyocr.Reader(['en'])

# you can use this below for no gpu devices
# reader = easyocr.Reader(['en'], gpu=False)

# Performing OCR on an image
image = 'test/2.jpg'
result = reader.readtext(image)
img = cv2.imread(image)

for result in result:
    print(result)
    top_left = tuple(result[0][0])
    bottom_right = tuple(result[0][2])
    text = re.sub("""[."'}]""","",result[1])
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.rectangle(img,top_left,bottom_right,(0,255,100),2)
    img = cv2.putText(img,text,bottom_right, font, 1.5,(0,255,0),4,cv2.LINE_AA)

# Instead of showing, save the figure
plt.figure(figsize=(10,10))
plt.imshow(img)
plt.axis('off')  # Remove axes for cleaner output
plt.savefig('output_image.jpg', bbox_inches='tight', pad_inches=0)
plt.close()  # Close the figure to free memory