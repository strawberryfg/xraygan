import pandas as pd

type_name = "Hernia"
train_path = "/content/drive/MyDrive/xray GAN/xray14/"
os.chdir(train_path)

type_df = "df_" + type_name
df = pd.read_csv(train_path + "data_entry.csv")
type_df = df[df['Finding Labels'] == type_name].copy()

type_files = open(train_path + type_name + "_files.txt", "w")
type_df['Image Index'].to_csv(type_files, sep="\n", index=None, header=None)
