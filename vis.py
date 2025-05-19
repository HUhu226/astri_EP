import pickle

# 文件路径
file_path = "/home/chesley/astri/AI4EDA-EfficientPlace/workspace/adaptec1/05-13/11:40:01/sol/best_placement_6.07834005355835.pkl"

# 加载文件内容
with open(file_path, "rb") as f:
    data = pickle.load(f)

# 打印内容类型和部分数据
print(type(data))
print(data)