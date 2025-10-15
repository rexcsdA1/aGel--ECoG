
from statsmodels.formula.api import glm
import statsmodels.api as sm
from scipy.stats import chi2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.io import loadmat
# 读取数据

# 假设当前文件夹有 example.mat
mat_data = loadmat('+aGel_Impedance.mat')
# 假设 'data' 是结构数组
data_struct = mat_data['data']

# 把结构体字段名取出来
fields = data_struct.dtype.names

# 转成 dict，注意索引
data_dict = {field: data_struct[field][0,0].flatten() for field in fields}

# 再转成 DataFrame
data = pd.DataFrame(data_dict)


breakpoint_count = 2
breakpoint1= 11
breakpoint2= 40
# 判断文件名后缀来决定是否进行分段拟合
if breakpoint_count == 2:


    # 根据 Implant_Day 分为三个部分
    data['Segment'] = np.select(
        [
            (data['Implant_Day'] < breakpoint1),  # 第一部分
            (data['Implant_Day'] >= breakpoint1) & (data['Implant_Day'] <= breakpoint2),  # 第二部分
            data['Implant_Day'] > breakpoint2,  # 第三部分
        ],
        [1, 2, 3],  # 对应的标签
        default=np.nan
    )

    # 获取Segment的天数范围
    segment_ranges = {
        1: (data['Implant_Day'].min(), breakpoint1),
        2: (breakpoint1, breakpoint2),
        3: (breakpoint2, data['Implant_Day'].max())
    }

    # 初始化拟合结果和颜色
    fit_results = []


    # 绘图
    plt.figure(figsize=(18, 9))

    # 绘制散点图，根据 Rat_ID 使用不同颜色
    for i, (rat_id, rat_data) in enumerate(data.groupby('Rat_ID')):
        plt.scatter(
            rat_data['Implant_Day'],
            rat_data['Impedance'],
            label=f'Rat {rat_id}',  # 使用统一的 Rat_ID 标签
            alpha=0.5,
            s=70
        )

    # 对每个分段进行拟合并计算斜率、p值、R²
    for segment in [1, 2, 3]:
        segment_data = data[data['Segment'] == segment]

        # 拟合广义线性模型（GLM）
        formula = 'Impedance ~ Implant_Day'  # 自变量为 Implant_Day，因变量为 Impedance(KΩ)
        model = glm(formula=formula, data=segment_data, family=sm.families.Gaussian()).fit()
        fit_results.append(model)

        # 受限模型（不包含 Implant_Day，仅截距项）
        model_null = glm(formula='Impedance ~ 1', data=segment_data, family=sm.families.Gaussian()).fit()

        # 计算 LRT 统计量
        lr_stat = -2 * (model_null.llf - model.llf)
        # 计算 p 值 (自由度=1)
        p_value = chi2.sf(lr_stat, df=1)  # Survival function (1 - CDF)
        formatted_p_value = "{:.2e}".format(p_value)
        # 打印拟合结果
        print(f"段 {segment} 的拟合结果：\n{model.summary()}")

        # 获取拟合数据的天数范围
        fitted_min_day = segment_data['Implant_Day'].min()
        fitted_max_day = segment_data['Implant_Day'].max()

        # 检查拟合数据是否覆盖了预期的 Implant_Day 范围
        expected_range = segment_ranges[segment]

        # 如果拟合数据的范围不符合预期，补充数据
        if fitted_min_day > expected_range[0]:
            missing_days = np.arange(expected_range[0], fitted_min_day)
            predicted_values = model.predict(pd.DataFrame({'Implant_Day': missing_days}))
            segment_data = pd.concat(
                [pd.DataFrame({'Implant_Day': missing_days, 'Impedance': predicted_values}), segment_data])

        if fitted_max_day < expected_range[1]:
            missing_days = np.arange(fitted_max_day + 1, expected_range[1] + 1)
            predicted_values = model.predict(pd.DataFrame({'Implant_Day': missing_days}))
            segment_data = pd.concat(
                [segment_data, pd.DataFrame({'Implant_Day': missing_days, 'Impedance': predicted_values})])

        # 重新绘制拟合线
        x_vals = np.linspace(segment_data['Implant_Day'].min(), segment_data['Implant_Day'].max(), 100)
        y_vals = model.predict(pd.DataFrame({'Implant_Day': x_vals}))
        valid_indices = y_vals > 0
        x_vals = x_vals[valid_indices]
        y_vals = y_vals[valid_indices]  # 拟合线使用不同的样式
        plt.plot(x_vals, y_vals, linestyle='-' if segment == 1 else '--', linewidth=2, label=f'Fit Segment {segment}')

        # 提取斜率、p值和伪R²
        slope = model.params['Implant_Day']
        p_value = model.pvalues['Implant_Day']
        print(f"段 {segment} 的拟合结果：")
        print(f"斜率: {slope:.2f}")
        print(f"LRT p 值: {formatted_p_value}")

        # 图例、标题和标签
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # 设置坐标轴线宽为 2
    plt.tick_params(axis='both', width=2, labelsize=25)
    plt.xticks(fontsize=25, family='Arial')
    plt.yticks(fontsize=25, family='Arial')
    # 设置坐标框的线宽为 2
    ax = plt.gca()  # 获取当前坐标轴
    ax.spines['top'].set_linewidth(2)  # 设置顶部框线宽
    ax.spines['right'].set_linewidth(2)  # 设置右侧框线宽
    ax.spines['left'].set_linewidth(2)  # 设置左侧框线宽
    ax.spines['bottom'].set_linewidth(2)  # 设置底部框线宽
    plt.xlabel('Days post-implantation', fontsize=25, family='Arial')
    plt.ylabel('Impedance (kΩ)', fontsize=25, family='Arial')
    # Display the grid

    plt.grid(True)

    # 设置字体属性
    font_properties = font_manager.FontProperties(family='Arial', size=25)

    legend = plt.legend(ncol=2, fontsize=18, prop=font_properties, frameon=False)
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)

    # 禁用网格线
    plt.grid(False)
    plt.show()

else:

    # 绘图
    plt.figure(figsize=(18, 9))

    # 绘制散点图，根据 Rat_ID 使用不同颜色
    for i, (rat_id, rat_data) in enumerate(data.groupby('Rat_ID')):
        plt.scatter(
            rat_data['Implant_Day'],
            rat_data['Impedance'],
            label=f'Rat {rat_id}',  # 使用统一的 Rat_ID 标签
            alpha=0.5,
            s=70
        )

    # 整体拟合广义线性模型（GLM）
    formula = 'Impedance ~ Implant_Day'
    model = glm(formula=formula, data=data, family=sm.families.Gaussian()).fit()

    # 受限模型（不包含 Implant_Day，仅截距项）
    model_null = glm(formula='Impedance ~ 1', data=data, family=sm.families.Gaussian()).fit()

    # 计算 LRT 统计量
    lr_stat = -2 * (model_null.llf - model.llf)
    # 计算 p 值 (自由度=1)
    p_value = chi2.sf(lr_stat, df=1)  # Survival function (1 - CDF)
    formatted_p_value = "{:.2e}".format(p_value)

    # 绘制拟合线
    x_vals = np.linspace(data['Implant_Day'].min(), data['Implant_Day'].max(), 100)
    y_vals = model.predict(pd.DataFrame({'Implant_Day': x_vals}))
    plt.plot(x_vals, y_vals, linestyle='-', linewidth=2, color='black', label='Fit line')

    # 提取斜率、p值
    slope = model.params['Implant_Day']
    p_value = model.pvalues['Implant_Day']
    # 打印统计信息
    print(f"拟合结果：")
    print(f"斜率: {slope:.2f}")
    print(f"p值: {p_value:.4f}")
    print(f"LRT p 值: {formatted_p_value}")
    print(x_vals)
    print(y_vals)
    plt.tick_params(axis='both', width=2, labelsize=25)
    plt.xticks(fontsize=25, family='Arial')
    plt.yticks(fontsize=25, family='Arial')
    # 设置坐标框的线宽为 2
    ax = plt.gca()  # 获取当前坐标轴
    ax.spines['top'].set_linewidth(2)  # 设置顶部框线宽
    ax.spines['right'].set_linewidth(2)  # 设置右侧框线宽
    ax.spines['left'].set_linewidth(2)  # 设置左侧框线宽
    ax.spines['bottom'].set_linewidth(2)  # 设置底部框线宽
    plt.xlabel('Days post-implantation', fontsize=25, family='Arial')
    plt.ylabel('Impedance (kΩ)', fontsize=25, family='Arial')
    # Display the grid

    plt.grid(True)

    # 设置字体属性
    font_properties = font_manager.FontProperties(family='Arial', size=25)

    legend = plt.legend(ncol=2, fontsize=18, prop=font_properties, frameon=False)
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)

    # 禁用网格线
    plt.grid(False)
    plt.show()
