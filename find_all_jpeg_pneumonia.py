import glob, os
# train
train_pneumonia_path = "E:/ml/ZhangLabData/CellData/chest_xray/train/PNEUMONIA/"
os.chdir(train_pneumonia_path)
pneumonia_files = open(train_pneumonia_path + "pneumonia_files.txt", "w")
for file in glob.glob("*.jpeg"):
    print(file)
    pneumonia_files.write(file)
    pneumonia_files.write('\n')
pneumonia_files.close()

# test
test_pneumonia_path = "E:/ml/ZhangLabData/CellData/chest_xray/test/PNEUMONIA/"
os.chdir(test_pneumonia_path)
pneumonia_files = open(test_pneumonia_path + "pneumonia_files.txt", "w")
for file in glob.glob("*.jpeg"):
    print(file)
    pneumonia_files.write(file)
    pneumonia_files.write('\n')
pneumonia_files.close()
