import glob, os
# train
train_normal_path = "E:/ml/ZhangLabData/CellData/chest_xray/train/NORMAL/"
os.chdir(train_normal_path)
normal_files = open(train_normal_path + "normal_files.txt", "w")
for file in glob.glob("*.jpeg"):
    print(file)
    normal_files.write(file)
    normal_files.write('\n')
normal_files.close()

# test
test_normal_path = "E:/ml/ZhangLabData/CellData/chest_xray/test/NORMAL/"
os.chdir(test_normal_path)
normal_files = open(test_normal_path + "normal_files.txt", "w")
for file in glob.glob("*.jpeg"):
    print(file)
    normal_files.write(file)
    normal_files.write('\n')
normal_files.close()
