import pandas as pd
train_path = "E:/ml/"
labels_name = "labels"
df = pd.read_csv(train_path + "data_entry.csv")
labels = df['Finding Labels']
labels_files = open(train_path + labels_name + ".txt", "w")
labels.to_csv(labels_files, sep="\n", index=None, header=None)

