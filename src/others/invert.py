import cv2

img_path = "C:/Users/BlackLufus/Workspace/portfolio_website/public/images/night.png"
output_path = "C:/Users/BlackLufus/Workspace/portfolio_website/public/images/night_new.png"

img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

if img is not None:
    if img.shape[2] == 4:
        b, g, r, a = cv2.split(img)

        rgb_invert = cv2.merge([255 - b, 255 - g, 255 - r])

        img_invert = cv2.merge([255 - b, 255 - g, 255 - r, a])

    else: 
        img_invert = 255 - img

    cv2.imwrite(output_path, img_invert)