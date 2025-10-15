import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pygam import GAM, s
from scipy.interpolate import interp1d
from matplotlib import font_manager
from scipy.io import loadmat
# 假设当前文件夹有 example.mat
mat_data = loadmat('-aGel_Impedance.mat')
# 假设 'data' 是结构数组
data_struct = mat_data['data']

# 把结构体字段名取出来
fields = data_struct.dtype.names

# 转成 dict，注意索引
data_dict = {field: data_struct[field][0,0].flatten() for field in fields}

# 再转成 DataFrame
data = pd.DataFrame(data_dict)
# 准备筛选数据
unique_electrodes = data['Electrode_ID'].unique()
rat_ids = data['Rat_ID'].unique()

# 准备拟合数据
X = data['Implant_Day'].values
y = data['Impedance'].values
rat_ids = data['Rat_ID'].values

# 创建绘图对象
plt.figure(figsize=(18,9))
plt.figure(1)


# 分组绘图
for rat_id in np.unique(rat_ids):
    rat_data = data[data['Rat_ID'] == rat_id]

    X_rat = rat_data['Implant_Day'].values
    y_rat = rat_data['Impedance'].values

    # GAM 模型拟合
    gam = GAM(s(0)).fit(X_rat, y_rat)

    # 预测趋势
    X_test = np.linspace(X_rat.min(), X_rat.max(), 1000)
    y_pred = gam.predict(X_test)



     # 绘制散点图，设置散点大小和透明度
    plt.scatter(X_rat, y_rat, alpha=0.5, s=70, label=f'{rat_id}')  # s=50 设置散点大小，alpha=0.7 设置透明度



gam_all = GAM(s(0)).fit(X, y)
X_full = np.linspace(X.min(), X.max(), 111)
y_all_pred = gam_all.predict(X_full)


# 保存预测值
predictions_df = pd.DataFrame({
    "Implant_Day": X_full,
    "Predicted_Impedance": y_all_pred
})


#均值加标准差
#提取大于等于 60 天的所有数据
data_60_days = data[data['Implant_Day'] >=60]['Impedance']

predicted_60_days = predictions_df[predictions_df["Implant_Day"] >= 60]["Predicted_Impedance"]


#上四分位数
# 提取大于等于 60 天的预测值
data_60_days = data[data['Implant_Day'] >= 60]['Impedance']
predicted_60_days = predictions_df[predictions_df["Implant_Day"] >= 60]["Predicted_Impedance"]


# 75% 分位数 Q3
q1= data_60_days.quantile(0.25)
q3 = data_60_days.quantile(0.75)
mean = predicted_60_days.mean()


# 计算 3/4 分位区间
IQR = q3-q1;
upper_bound = mean +IQR  # 下界
lower_bound = mean - IQR  # 上界


# 在预测曲线图上绘制四分位数区间
plt.plot(X_full, y_all_pred, color='red', label='Fitting line', linewidth=2)
plt.axhline(mean, color='blue', linestyle='--', label='mean (after 60 days)', linewidth=2)
plt.fill_between(
    [10, X_full.max()],
    lower_bound,
    upper_bound,
    color='blue',
    alpha=0.2,
    label='mean ± IQR'
)
# 找到拟合曲线的极大值处
max_value_idx = np.argmax(y_all_pred)
max_value_day = X_full[max_value_idx]
max_value_day_rounded = int(np.ceil(max_value_day))
max_value_Im = y_all_pred[max_value_idx]
print(max_value_day_rounded)
# 在极大值处画一条垂直的黑色虚线
# plt.axvline(x=max_value_day, color='black', linestyle='-', label=f'Break-points 1 at Day 11 ({max_value_day:.2f})', linewidth=2)
plt.axvline(x=max_value_day, color='black', linestyle='-',linewidth=2)
plt.text(35, 700, f'breakpoint 1 = day {max_value_day_rounded }', color='black',
         horizontalalignment='right', verticalalignment='bottom', fontsize=19, family='Arial')


# 使用插值方法构造函数来寻找 X_full 中 y_all_pred 与 upper_bound_all 的交点
interp_func = interp1d(y_all_pred, X_full, kind='linear', fill_value='extrapolate')

# 计算拟合曲线与 upper_bound 的交点
if upper_bound >= y_all_pred.min() and upper_bound <= y_all_pred.max():
    intersection_day = interp_func(upper_bound)
    print(f"拟合曲线与上界 ({upper_bound:.4f} kΩ) 的交点在 Implant_Day: {intersection_day:.2f} 天")
    intersection_day_rounded = int(np.ceil(intersection_day))
    # 在交点处画一条 x 轴的黑色实线
    # plt.axvline(x=intersection_day, color='black', linestyle='--', label=f'Break-points 2 at Day 32 ({intersection_day:.2f})', linewidth=2)
    # 在交点处画一条 x 轴的黑色实线
    plt.axvline(x=intersection_day, color='black', linestyle='-',linewidth=2)
    plt.text(70, 450, f'breakpoint 2 = day {intersection_day_rounded}', color='black',
             horizontalalignment='right', verticalalignment='bottom', fontsize=19, family='Arial')
    # 在交点处添加箭头指向交点
    plt.annotate(
        '',
        xy=(intersection_day, 91.75),  # 箭头的目标位置（交点）
        xytext=(55, 450),  # 箭头的起始位置
        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=10)
    )
else:
    print("上界与拟合曲线没有交点，可能是数据范围问题。")

# 在极大值处添加箭头指向极大值点
plt.annotate(
    '',
    xy=(max_value_day, y_all_pred.max() ),  # 极大值点位置
    xytext=(25, 700),  # 箭头的起始位置
    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=10)
)

# 设置图表样式
# 设置字体为 Arial

# 设置坐标轴线宽为 2
plt.tick_params(axis='both',width=2, labelsize=25)
plt.xticks(fontsize=25, family='Arial')
plt.yticks(fontsize=25, family='Arial')
# 设置坐标框的线宽为 2
ax = plt.gca()  # 获取当前坐标轴
ax.spines['top'].set_linewidth(2)  # 设置顶部框线宽
ax.spines['right'].set_linewidth(2)  # 设置右侧框线宽
ax.spines['left'].set_linewidth(2)  # 设置左侧框线宽
ax.spines['bottom'].set_linewidth(2)  # 设置底部框线宽
plt.xlabel('Days post-implantation', fontsize=25,family = 'Arial')
plt.ylabel('Impedance (kΩ)', fontsize=25,family = 'Arial')
# Display the grid

plt.grid(True)


# 设置字体属性
font_properties = font_manager.FontProperties(family='Arial', size=25)


legend = plt.legend(ncol=2, fontsize=18, prop=font_properties, frameon=False)
for handle in legend.legend_handles:
    handle.set_alpha(1.0)

# 禁用网格线
plt.grid(False)

# 保存图片到指定位置，dpi=600
# plt.savefig('V:/临时/沈阳/file/data/your_image.png', dpi=1000)

# 显示图形
plt.show()
