import pandas as pd
import numpy as np

print("正在生成测试数据...")

# --- 1. 生成干净的标准数据 (CSV) ---
# 模拟：根据 GDP 和 能源消耗(Energy) 预测 碳排放(CO2)
# 年份(Year) 是索引列
data_clean = pd.DataFrame({
    'Year': range(2000, 2025),  # 2000-2024
    'GDP': np.linspace(10, 50, 25) + np.random.normal(0, 1, 25), # 线性增长+噪声
    'Energy': np.linspace(50, 100, 25) + np.random.normal(0, 2, 25),
    'CO2': np.linspace(200, 400, 25) + np.random.normal(0, 5, 25) # 目标变量
})
data_clean.to_csv('test_clean.csv', index=False)
print("✅ 已生成: test_clean.csv (标准数据)")

# --- 2. 生成带杂质的 Excel 数据 (XLSX) ---
# 这一份数据里多了一列 'Comments' (字符串)，这是PyMCM应该报错的
data_dirty = data_clean.copy()
data_dirty['Comments'] = ['Normal'] * 24 + ['Peak Year'] # 加了一列文字
data_dirty.to_excel('test_dirty.xlsx', index=False)
print("✅ 已生成: test_dirty.xlsx (含脏数据)")