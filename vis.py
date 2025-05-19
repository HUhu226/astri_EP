import pickle

# 文件路径
file_path = "/home/chesley/astri/AI4EDA-EfficientPlace/workspace/adaptec1/05-13/11:40:01/sol/best_placement_6.07834005355835.pkl"

# 加载文件内容
with open(file_path, "rb") as f:
    data = pickle.load(f)

# 打印内容类型和部分数据
print(type(data))
print(data)



# 访问数据中的特定字段
# 这里假设数据是一个字典，包含了你需要的字段
# 你可以根据实际数据结构进行修改
# 例如，假设数据包含一个名为 'placement' 的字段
# if 'placement' in data:
#     placement = data['placement']
#     print("Placement data:", placement)