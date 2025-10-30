import os

label = "0"

search_pattern = label + "_"

dir_path = "data/old_train"

for file in os.listdir(dir_path):
    # if file.startswith(search_pattern):
    label = file[0]
    target_dir_path = os.path.join("data/train", label)

    filename = f"{len(os.listdir(target_dir_path)):04d}.png"
    print(os.path.join(dir_path, file))
    print(os.path.join(target_dir_path, filename))
    os.replace(os.path.join(dir_path, file), os.path.join(target_dir_path, filename))
    # break
