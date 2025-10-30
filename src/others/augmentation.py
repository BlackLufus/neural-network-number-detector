import os

import cv2

src_dir_path = "data"
target_dir_path = "data/train"

rotations = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE
}

for label in range(10):

    path_to_folder = os.path.join(src_dir_path, str(label))

    for image in os.listdir(path_to_folder):

        path_to_src_file = os.path.join(path_to_folder, image)
        img = cv2.imread(path_to_src_file, cv2.IMREAD_GRAYSCALE)
    
        for angle, rotate_code in rotations.items():
            if rotate_code is None:
                rotate_img = img.copy()  # 0 Grad -> Original
            else:
                rotate_img = cv2.rotate(img, rotateCode=rotate_code)
            path_to_target_folger = os.path.join(target_dir_path, str(label))
            os.makedirs(path_to_target_folger, exist_ok=True)
            path_to_target_file = os.path.join(path_to_target_folger, f"{len(os.listdir(path_to_target_folger)):04d}.png")
            cv2.imwrite(path_to_target_file, rotate_img)

            print(f"Save file: {path_to_target_file}")
