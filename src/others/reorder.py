import os

dir_path = "data/train/"

print("Test")

for label in range(10):

    id = 0
    print(f"Elements: {len(os.listdir(dir_path))}")
    
    dir_to_label = os.path.join(dir_path, str(label))
    for file in os.listdir(dir_to_label):
        new_file = filename = f"{id:04d}.png"
        id += 1
        print(f"{os.path.join(dir_to_label, file)} -> {os.path.join(dir_to_label, new_file)}")
        os.replace(os.path.join(dir_to_label, file), os.path.join(dir_to_label, new_file))