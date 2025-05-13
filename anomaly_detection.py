import random
import csv
import matplotlib.pyplot as plt

#从CSV文件中读取数据
def read_csv_and_filter_by_label(file_path, target_label):
    content = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            label = int(row[-1])  # 标签是最后一列
            if label == target_label:
                image_data = list(map(float, row[:-1]))  # 前 784 个像素
                content.append(image_data)
    return content


# 2. 利用3-sigma方法分析数据中的异常值
def detect_anomalies(images):
    # 将所有图像的像素值拼接为一个一维数组用于统计分析
    all_pixels = [pixel for image in images for pixel in image]

    # 计算均值和标准差
    mean = sum(all_pixels)/len(all_pixels)
    variance = sum((x-mean)**2 for x in all_pixels)/len(all_pixels)
    std=variance ** 0.5

    # 通过 3-sigma 方法找出异常值
    lower_bound = mean-3 *std
    upper_bound = mean+3 *std

    # 标记异常数据及其异常程度
    anomalies = []
    for image in images:
        # 计算该图像的异常程度，计算所有像素的异常值
        anomaly_score = sum(1 for pixel in image if pixel < lower_bound or pixel > upper_bound)
        anomalies.append((image, anomaly_score))
    
    # 根据异常程度排序，得分高的在前
    anomalies.sort(key=lambda x: x[1], reverse=True)
    
    return anomalies

# 3. 可视化前10个最异常数据（如果少于10个，就显示全部）
def visualize_anomalies(anomalies):
    num_to_show = min(10, len(anomalies))  # 如果异常数据少于10个，就显示所有
    for i in range(num_to_show):
        image, _ = anomalies[i]
        # 将一维列表转为28x28的二维列表进行显示
        image_2d = [image[i:i+28] for i in range(0, 784, 28)]
        plt.imshow(image_2d, cmap='gray')  # 直接传入二维列表
        plt.title(f"Anomaly {i+1}")
        plt.axis('off')  # 关闭坐标轴
        plt.show()

# 设置文件路径和目标标签
csv_file = 'mnist_test.csv'
target_label = 5  # 例如，标签为5的数字

# 1. 从CSV文件中读取目标标签的数据
images = read_csv_and_filter_by_label(csv_file, target_label)

# 2. 检测异常数据并排序
anomalies = detect_anomalies(images)
print(len(anomalies))

# 3. 可视化最异常的前10个数据
visualize_anomalies(anomalies)
