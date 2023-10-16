import os
def data_split(path,num):
    root = os.path.join(path,num)
    with open("dataset/doc3d/train/train_1.txt", "a") as f:
        for file in os.listdir(root):
            file_name = file.split('.mat')[0]
            file_name = f'{num}/{file_name}'+'\n'
            f.writelines(file_name)

if __name__=="__main__":
    data_split("dataset/doc3d/train/bm/", num='5')