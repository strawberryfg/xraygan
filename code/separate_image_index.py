import pandas as pd
train_path = "E:/ml/"
image_index_name = "image_index"
df = pd.read_csv(train_path + "data_entry.csv")
image_index = df['Image Index']
image_index_files = open(train_path + image_index_name + ".txt", "w")
image_index.to_csv(image_index_files, sep="\n", index=None, header=None)