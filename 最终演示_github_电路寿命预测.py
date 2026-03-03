#####

#####


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from SSA import SSA
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from scipy import stats, optimize
from scipy.stats import lognorm
from scipy.optimize import fsolve,minimize
import scipy.optimize as opt
from bayes_opt import BayesianOptimization
import time
from scipy.optimize import minimize, Bounds
from pathlib import Path
if 'first_part_completed' not in st.session_state:
    st.session_state['first_part_completed'] = False
# Streamlit 应用标题
st.title('📟电路可靠性预测')
st.subheader("一、数据选取")
# 文件上传
# 文件上传（可选）+ 默认加载仓库示例文件
file_path = st.sidebar.file_uploader("上传 Excel 文件（可选：上传后将覆盖示例数据）", type=['xlsx'])

# ✅ 默认示例文件路径：data/demo.xlsx（相对 app.py 所在目录）
DEMO_EXCEL = Path(__file__).resolve().parent / "测试数据.xlsx"

if file_path is None:
    if DEMO_EXCEL.exists():
        file_path = str(DEMO_EXCEL)  # 让后面代码保持不变（仍然用 file_path）
        st.sidebar.caption(f"当前数据源：{DEMO_EXCEL.name}（示例文件）")
    else:
        st.sidebar.error(f"未找到示例文件：{DEMO_EXCEL}。请上传 Excel 或将 demo.xlsx 放到 data/ 目录。")
        st.stop()
else:
    st.sidebar.caption(f"当前数据源：{file_path.name}（已上传）")


if file_path is not None:
    try:
        # 加载 Excel 文件的工作簿名称
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names

        # 让用户选择一个工作簿
        sheet_name = st.sidebar.selectbox("选择一个工作簿", sheet_names)

        # 添加模型选择器
        # 模型选择
        model_options = {
            "肖特基势垒二极管金半接触退化模型": "Schottky Barrier Diode Gold-Semiconductor Degradation Model",
            "肖特基势垒二极管金属化电迁移模型": "Schottky Barrier Diode Electromigration Model",
            "PN结整流二极管PN结特性退化模型": "PN Junction Rectifier Diode PN Junction Degradation Model",
            "PN结整流二极管热载流子注入模型": "PN Junction Rectifier Diode Hot Carrier Injection Model",
            "双极晶体管热载流子注入模型": "Bipolar Transistor Hot Carrier Injection Model",
            "双极晶体管PN结特性退化模型": "Existing Model"
        }
        selected_model = st.selectbox("选择一个模型", list(model_options.keys()))
        # 根据选择的模型调整图表标题和模型公式
        # 根据选择的模型调整图表标题和模型公式
        if selected_model == "肖特基势垒二极管金半接触退化模型":
            model_formula = r'''
                    $$
                    V_F = V_{F0} + A \times V_R^m \times e^{\frac{{-E_a}}{{K \times T}}} \times t^p
                    $$
                    '''
            ylabel = '∆V_F'
            chart_title = model_options[selected_model]

        elif selected_model == "肖特基势垒二极管金属化电迁移模型":
            model_formula = r'''
                    $$
                    V_F = V_{F0} + A \times I_R^n \times e^{\frac{{-E_a}}{{R \times T}}} \times t^p
                    $$
                    '''
            ylabel = '∆V_F'
            chart_title = model_options[selected_model]

        elif selected_model == "PN结整流二极管PN结特性退化模型":
            model_formula = r'''
                    $$
                    I_R = I_{R0} + A \times V_R^m \times e^{\frac{{-E_a}}{{k \times T}}} \times t^p
                    $$
                    '''
            ylabel = '∆I_R'
            chart_title = model_options[selected_model]

        elif selected_model == "PN结整流二极管热载流子注入模型":
            model_formula = r'''
                    $$
                    I_R = I_{R0} + A \times I_F^n \times e^{\frac{{-E_a}}{{k \times T}}} \times t^p
                    $$
                    '''
            ylabel = '∆I_R'
            chart_title = model_options[selected_model]

        elif selected_model == "双极晶体管热载流子注入模型":
            model_formula = r'''
                    $$
                    \beta = \beta_0 + A \times I_c^n \times e^{\frac{{-E_a}}{{k \times T}}} \times t^p
                    $$
                    '''
            ylabel = '∆β'
            chart_title = model_options[selected_model]

        else:
            model_formula = r'''
                    $$
                    \beta = \beta_0 + A \times (\lvert V_R \rvert^m) \times e^{\frac{{-E_a}}{{K \times T}}} \times (t^p)
                    $$
                    '''
            ylabel = '∆β'
            chart_title = "数据分析和拟合"

        # 在 Streamlit 应用中显示选定的模型公式
        st.write("模型公式：")
        st.markdown(model_formula, unsafe_allow_html=True)

        # 读取选定的工作簿数据
        df = xls.parse(sheet_name)

        with st.expander("**数据在线编辑：**"):
            # 显示 DataFrame
            # 使用 AgGrid 创建可编辑的表格
            grid = AgGrid(
                df,
                editable=True,  # 启用编辑功能
                height=400,  # 设置表格高度
                width='100%',  # 设置表格宽度为100%

            )

            # 获取编辑后的 DataFrame
            updated_df = grid['data']
        st.write("编辑后数据：")
        st.dataframe(updated_df)
        for column in updated_df.columns:
            # 尝试将每一列转换为数值类型
            updated_df[column] = pd.to_numeric(updated_df[column], errors='coerce')
        #st.write(updated_df.dtypes)

        #st.write('updated_df', updated_df)
        df = updated_df.copy()

        #st.write('df', df)
        #st.dataframe(df)

    except Exception as e:
        st.error(f"处理文件时发生错误：{e}")
    # 使用 Streamlit 的交互性组件来获取用户输入
    st.sidebar.header("设置输入参数")
    # 添加复选框和条件性数值输入

    # 用户输入行和列范围的示例格式："起始行:结束行, 起始列:结束列"
    # 使用 Streamlit 的交互性组件来分组设置应力参数
    with st.sidebar.expander("应力条件1的数据范围"):
        range_input_1 = st.text_input("条件1数据范围", "2:11, 3:11", help="格式为: 起始行:结束行, 起始列:结束列")
        T_C_1 = st.number_input("应力1.1", value=398.000, format="%.4f",help="开尔文温度")
        VR1 = st.number_input("应力1.2", value=200.000, format="%.4f")

    with st.sidebar.expander("应力条件2的数据范围"):
        range_input_2 = st.text_input("条件2数据范围", "12:21, 3:11", help="格式为: 起始行:结束行, 起始列:结束列")
        T_C_2 = st.number_input("应力2.1", value=398.000, format="%.4f",help="都是卡尔文温度")
        VR2 = st.number_input("应力2.2", value=600.000, format="%.4f")

    with st.sidebar.expander("应力条件3的数据范围"):
        range_input_3 = st.text_input("条件3数据范围", "22:31, 3:11", help="格式为: 起始行:结束行, 起始列:结束列")
        T_C_3 = st.number_input("应力3.1", value=398.000, format="%.4f")
        VR3 = st.number_input("应力3.2", value=600.000, format="%.4f")
    with st.sidebar.expander("应力条件4的数据范围"):
        range_input_4 = st.text_input("条件4数据范围", "36:45, 3:18", help="格式为: 起始行:结束行, 起始列:结束列")
        T_C_4 = st.number_input("应力4.1", value=423.000, format="%.4f")
        VR4 = st.number_input("应力4.2", value=600.000, format="%.4f")
    def parse_range(range_str):
        # 解析行和列范围
        row_range, col_range = range_str.split(',')
        start_row, end_row = map(int, row_range.split(':'))
        start_col, end_col = map(int, col_range.split(':'))
        return start_row, end_row, start_col, end_col
    # 解析用户输入的范围
    start_row_1, end_row_1, start_col_1, end_col_1 = parse_range(range_input_1)
    start_row_2, end_row_2, start_col_2, end_col_2 = parse_range(range_input_2)
    start_row_3, end_row_3, start_col_3, end_col_3 = parse_range(range_input_3)
    start_row_4, end_row_4, start_col_4, end_col_4 = parse_range(range_input_4)

    # 为四组 t 数据设置行和列范围的输入
    with st.expander("应力条件1时间"):
        t1_row_index = st.number_input("t1 数据所在的行索引", value=1, min_value=0)
        t1_start_col_index = st.number_input("t1 数据的起始列索引", value=3, min_value=0)
        t1_end_col_index = st.number_input("t1 数据的结束列索引", value=11, min_value=0)

    with st.expander("应力条件2时间"):
        t2_row_index = st.number_input("t2 数据所在的行索引", value=1, min_value=0)
        t2_start_col_index = st.number_input("t2 数据的起始列索引", value=3, min_value=0)
        t2_end_col_index = st.number_input("t2 数据的结束列索引", value=11, min_value=0)

    with st.expander("应力条件3时间"):
        t3_row_index = st.number_input("t3 数据所在的行索引", value=1, min_value=0)
        t3_start_col_index = st.number_input("t3 数据的起始列索引", value=3, min_value=0)
        t3_end_col_index = st.number_input("t3 数据的结束列索引", value=11, min_value=0)

    with st.expander("应力条件4时间"):
        t4_row_index = st.number_input("t4 数据所在的行索引", value=35, min_value=0)
        t4_start_col_index = st.number_input("t4 数据的起始列索引", value=3, min_value=0)
        t4_end_col_index = st.number_input("t4 数据的结束列索引", value=18, min_value=0)

    st.subheader("二、数据处理")
    def preprocess_data(beta_data, alpha):
        # 计算 IQR
        Q1 = beta_data.quantile(0.25)
        Q3 = beta_data.quantile(0.75)
        IQR = Q3 - Q1
        # 定义异常值的范围
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # 过滤异常值
        filtered_df = beta_data.where(~((beta_data < lower_bound) | (beta_data > upper_bound)))
        # 将 NaN 值替换为每列的平均值
        filtered_df = filtered_df.fillna(filtered_df.mean())
        # 数据平滑处理
        smoothed_df = filtered_df.ewm(alpha=alpha).mean()
        return filtered_df, smoothed_df

    # 使用 Streamlit 的侧边栏创建一个滑块来调节 alpha 值
    alpha = st.number_input("输入平滑系数 alpha", min_value=0.0, max_value=1.0, value=0.6, step=0.1,help="自动使用四分位数法（IQR）进行异常值剔除，此处使用指数加权平均法进行数据降噪处理")

    beta_1 = df.iloc[start_row_1:end_row_1 + 1, start_col_1:end_col_1 + 1]
    beta_2 = df.iloc[start_row_2:end_row_2 + 1, start_col_2:end_col_2 + 1]
    beta_3 = df.iloc[start_row_3:end_row_3 + 1, start_col_3:end_col_3 + 1]
    beta_4 = df.iloc[start_row_4:end_row_4 + 1, start_col_4:end_col_4 + 1]
    beta_1_0 = beta_1
    beta_2_0 = beta_2
    beta_3_0 = beta_3
    beta_4_0 = beta_4
    first_column1 = beta_1.iloc[:, 0]
    first_column2 = beta_2.iloc[:, 0]
    first_column3 = beta_3.iloc[:, 0]
    first_column4 = beta_4.iloc[:, 0]

    # 计算 Δbeta_1：每一列减去第一列
    #beta_1_1是求的差值
    beta_1_1 = beta_1.apply(lambda col: col - first_column1)
    beta_2_2 = beta_2.apply(lambda col: col - first_column2)
    beta_3_3 = beta_3.apply(lambda col: col - first_column3)
    beta_4_4 = beta_4.apply(lambda col: col - first_column4)


    #st.write('beta_1', beta_1)


    #st.write('beta_1',beta_1)

    beta_1_filtered, beta_1_smoothed = preprocess_data(beta_1, alpha)
    beta_2_filtered, beta_2_smoothed = preprocess_data(beta_2, alpha)
    beta_3_filtered, beta_3_smoothed = preprocess_data(beta_3, alpha)
    beta_4_filtered, beta_4_smoothed = preprocess_data(beta_4, alpha)

    # 获取第一列作为基准列
    smoothed1_first_column = beta_1_smoothed.iloc[:, 0]
    smoothed2_first_column = beta_2_smoothed.iloc[:, 0]
    smoothed3_first_column = beta_3_smoothed.iloc[:, 0]
    smoothed4_first_column = beta_4_smoothed.iloc[:, 0]
    # 计算 Δbeta_1：每一列减去第一列
    smoothed1 = beta_1_smoothed.apply(lambda col: col - smoothed1_first_column)
    smoothed2 = beta_2_smoothed.apply(lambda col: col - smoothed2_first_column)
    smoothed3 = beta_3_smoothed.apply(lambda col: col - smoothed3_first_column)
    smoothed4 = beta_4_smoothed.apply(lambda col: col - smoothed4_first_column)
    # 合并数据
    smoothed_data = {
        'stress_1_smoothed': beta_1_smoothed,
        'stress_2_smoothed': beta_2_smoothed,
        'stress_3_smoothed': beta_3_smoothed,
        'stress_4_smoothed': beta_4_smoothed
    }
    combined_df = pd.concat(smoothed_data,axis=1)

    # 转换为CSV
    csv = combined_df.to_csv(index=False).encode('utf-8')

    # 添加下载按钮
    st.download_button(
        label="下载数据",
        data=csv,
        file_name="smoothed_data.csv",
        mime="text/csv",
        help="点击此按钮下载处理后的数据"
    )


    # 提取四组 t 数据
    t1 = df.iloc[t1_row_index, t1_start_col_index:t1_end_col_index + 1].to_numpy()
    t2 = df.iloc[t2_row_index, t2_start_col_index:t2_end_col_index + 1].to_numpy()
    t3 = df.iloc[t3_row_index, t3_start_col_index:t3_end_col_index + 1].to_numpy()
    t4 = df.iloc[t4_row_index, t4_start_col_index:t4_end_col_index + 1].to_numpy()

    #st.write(t)
    #st.write(t)
    #st.write('beta_1_smoothed',beta_1_smoothed)
    # 选择框，让用户选择要显示的数据类型
    st.write('**预处理后结果：**')
    option = st.selectbox(
        '选择要显示的数据集',
        ('原始数据和平滑后的数据对比', '原始数据', '平滑后的数据'))
    # 用户选择是否显示图例
    # show_legend = st.checkbox("显示图例", value=True)
    # 绘制图形
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    # 根据用户选择的数据类型进行绘制
    if option == '原始数据':
        axs[0, 0].plot(t1, beta_1_1.T, 'o',)
        axs[0, 1].plot(t2, beta_2_2.T, 'o', )
        axs[1, 0].plot(t3, beta_3_3.T, 'o', )
        axs[1, 1].plot(t4, beta_4_4.T, 'o', )
    elif option == '平滑后的数据':
        axs[0, 0].plot(t1, smoothed1.T, 'o-', alpha=0.8)
        axs[0, 1].plot(t2, smoothed2.T, 'o-', alpha=0.8)
        axs[1, 0].plot(t3, smoothed3.T, 'o-', alpha=0.8 )
        axs[1, 1].plot(t4, smoothed4.T, 'o-', alpha=0.8)
    elif option == '原始数据和平滑后的数据对比':
        axs[0, 0].plot(t1, beta_1_1.T, 'o',)
        axs[0, 0].plot(t1, smoothed1.T, '-',)
        axs[0, 1].plot(t2, beta_2_2.T, 'o', label='2',alpha=0.6)
        axs[0, 1].plot(t2, smoothed2.T, '-', )
        axs[1, 0].plot(t3, beta_3_3.T, 'o', label='22',alpha=0.6)
        axs[1, 0].plot(t3, smoothed3.T, '-', )
        axs[1, 1].plot(t4, beta_4_4.T, 'o' , label='2',alpha=0.6)
        axs[1, 1].plot(t4, smoothed4.T, '-', )
    # 设置图例和图形标题
    for ax in axs.flat:
        #if show_legend:
           # ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel(ylabel)
        #ax.set_title(f'{chart_title} ')
        #ax.grid(True)
    # 设置子图标题
    axs[0, 0].set_title('Stress condition 1 ')
    axs[0, 1].set_title('Stress condition 2 ')
    axs[1, 0].set_title('Stress condition 3 ')
    axs[1, 1].set_title('Stress condition 4 ')
    # 调整布局
    plt.tight_layout()
    # 显示图形
    st.pyplot(fig)

    # 具体应力的输入1-4
    beta_1 = beta_1_smoothed.to_numpy()
    beta_2 = beta_2_smoothed.to_numpy()
    beta_3 = beta_3_smoothed.to_numpy()
    beta_4 = beta_4_smoothed.to_numpy()

    #st.write('beta_1',beta_1)
    #st.write(beta_175)


    # Beta_0 列索引
    # 使用 Streamlit 的交互性组件来分组设置 Beta_0 的列索引
    with st.sidebar.expander("设置t=0列索引（初始时刻的数据）"):
        # 让用户输入 beta_0_150 所在的列索引
        beta_0_1_col_index = st.number_input("应力1 t=0所在的列索引", value=3, min_value=0)

        # 让用户输入 beta_0_175 所在的列索引
        beta_0_2_col_index = st.number_input("应力2 t=0所在的列索引", value=3, min_value=0)
        beta_0_3_col_index = st.number_input("应力3 t=0所在的列索引", value=3, min_value=0)
        beta_0_4_col_index = st.number_input("应力4 t=0所在的列索引", value=3, min_value=0)
    # 根据用户输入的列索引获取 beta_0 值
    beta_0_1 = df.iloc[start_row_1:end_row_1 + 1, beta_0_1_col_index].mean()
    beta_0_2 = df.iloc[start_row_2:end_row_2 + 1, beta_0_2_col_index].mean()
    beta_0_3 = df.iloc[start_row_3:end_row_3 + 1, beta_0_3_col_index].mean()
    beta_0_4 = df.iloc[start_row_4:end_row_4 + 1, beta_0_4_col_index].mean()

    #beta_0=(beta_0_1+beta_0_2+beta_0_3+beta_0_4)/4


    #st.write('beta_0' , beta_0)
    #st.write(beta_0_175)
    # 将摄氏温度转换为开尔文
    T_1 = T_C_1
    T_2 = T_C_2
    T_3 = T_C_3
    T_4 = T_C_4
    # 根据用户输入的索引获取 t 数据

    # 玻尔兹曼常数，单位 eV/K
    K = 8.617333262145e-5

    formula_type = st.radio("选择所用数据的类型", ("正常数据", "增量型数据"),help=("增量型数据：指的是每个数据都跟所选数据的第一列（0时刻的初始值）做差"))

    if formula_type == "增量型数据":
        beta_1=smoothed1
        beta_2=smoothed2
        beta_3=smoothed3
        beta_4=smoothed4

        beta_0_1=0
        beta_0_2=0
        beta_0_3=0
        beta_0_4=0
    else:
        if selected_model == "肖特基势垒二极管金半接触退化模型":
            ylabel = 'V_F'

        elif selected_model == "肖特基势垒二极管金属化电迁移模型":
            ylabel = 'V_F'
            chart_title = model_options[selected_model]

        elif selected_model == "PN结整流二极管PN结特性退化模型":
            ylabel = 'I_R'

        elif selected_model == "PN结整流二极管热载流子注入模型":
            ylabel = 'I_R'

        elif selected_model == "双极晶体管热载流子注入模型":
            ylabel = 'β'

        else:
            ylabel = 'β'


    st.subheader("三、元器件可靠性建模")
    st.markdown("#### 3.1、定义搜索空间")
    fix_ea = st.checkbox('固定Ea')
    if fix_ea:
        ea_value = st.number_input("输入 E_a 的值", value=0.50, min_value=0.00, max_value=10.00, step=0.01,
                                           format="%.2f")
    # 使用 Streamlit 的交互性组件来分组设置参数
    with st.expander("设置 A 的搜索范围"):
        a_min = st.number_input("A 的最小值", value=-1000.00, format="%.2f")
        a_max = st.number_input("A 的最大值", value=1000.00, format="%.2f")

    with st.expander("设置 m 的搜索范围"):
        m_min = st.number_input("m 的最小值", value=2.00, min_value=0.00, max_value=10.00, step=0.10, format="%.2f")
        m_max = st.number_input("m 的最大值", value=4.00, min_value=0.00, max_value=10.00, step=0.10, format="%.2f")
    if not fix_ea:
        with st.expander("设置 E_a 的搜索范围"):
            ea_min = st.number_input("E_a 的最小值", value=0.10, min_value=0.00, max_value=10.00, step=0.10,
                                     format="%.2f")
            ea_max = st.number_input("E_a 的最大值", value=1.00, min_value=0.00, max_value=10.00, step=0.10,
                                     format="%.2f")

    with st.expander("设置 p 的搜索范围"):
        p_min = st.number_input("p 的最小值", value=0.50, min_value=0.00, max_value=10.00, step=0.10, format="%.2f")
        p_max = st.number_input("p 的最大值", value=2.00, min_value=0.00, max_value=10.00, step=0.10, format="%.2f")

    if fix_ea:
        # 固定 E_a 时的搜索空间，只包括 A、m 和 p
        E_a = ea_value
        search_space = {
            0: (a_min, a_max),  # A 的搜索范围
            1: (m_min, m_max),  # m 的搜索范围
            2: (p_min, p_max)  # p 的搜索范围
        }
        n_dim = 3  # 参数数量减少为 3
        def objective_function(params, beta_0, T, VR, beta, t):
            A, m,  p = params
            predicted_beta = beta_0 + A * (VR ** m) * np.exp(-(E_a / (K * T))) * (t ** p)
            mse = np.mean((beta - predicted_beta) ** 2)
            return mse

        def function(params, beta_0, T, VR, t):
            A, m, p = params
            predicted_beta = beta_0 + A * (np.abs(VR) ** m) * np.exp(-(E_a / (K * T))) * (t ** p)
            return predicted_beta

        def fitness_function(params):
            loss_1 = objective_function(params, beta_0_1, T_1, VR1, beta_1, t1)
            loss_2 = objective_function(params, beta_0_2, T_2, VR2, beta_2, t2)
            loss_3 = objective_function(params, beta_0_3, T_3, VR3, beta_3, t3)
            loss_4 = objective_function(params, beta_0_4, T_4, VR4, beta_4, t4)
            total_loss = loss_1 + loss_2 + loss_3 + loss_4
            return total_loss  # 在最小化问题中，适应度越低越好
    else:

        n_dim = 4
        # 根据用户输入定义搜索空间
        search_space = {
            0: (a_min, a_max),  # A 的搜索范围
            1: (m_min, m_max),  # m 的搜索范围
            2: (ea_min, ea_max),  # E_a 的搜索范围
            3: (p_min, p_max)  # p 的搜索范围
        }
        def objective_function(params, beta_0, T, VR, beta, t):
            A, m, E_a, p = params
            predicted_beta = beta_0 + A * (VR ** m) * np.exp(-(E_a / (K * T))) * (t ** p)
            mse = np.mean((beta - predicted_beta) ** 2)
            return mse

        def function(params, beta_0, T, VR, t):
            A, m, E_a, p = params
            predicted_beta = beta_0 + A * (np.abs(VR) ** m) * np.exp(-(E_a / (K * T))) * (t ** p)

            return predicted_beta

        def fitness_function(params):
            loss_1 = objective_function(params, beta_0_1, T_1, VR1, beta_1, t1)
            loss_2 = objective_function(params, beta_0_2, T_2, VR2, beta_2, t2)
            loss_3 = objective_function(params, beta_0_3, T_3, VR3, beta_3, t3)
            loss_4 = objective_function(params, beta_0_4, T_4, VR4, beta_4, t4)
            total_loss = loss_1 + loss_2 + loss_3 + loss_4
            return total_loss  # 在最小化问题中，适应度越低越好

    with st.sidebar.expander("均值和标准差（生成正态分布）"):
            # 使用st.sidebar.number_input来接收用户输入的均值和标准差
            mu = st.number_input("请输入均值:", value=0.91)
            sd = st.number_input("请输入标准差:", value=0.02)
            st.caption("说明：μ/σ 用于生成器件初值样本（Monte Carlo），用于计算寿命分布与 R(t)、F(t)。")

    st.sidebar.header("算法参数设置")
    # 让用户调整搜索算法中的个体数目，影响搜索的广度和速度
    pop_size = st.sidebar.number_input("搜索个体数目 (种群大小)",
                                 min_value=10,
                                 max_value=2000,
                                 value=100,
                                 help="调整个体数目来影响算法的搜索范围和速度。较大的数目可能提高找到最优解的概率，但会增加计算量。")

    # 让用户调整算法运行的迭代次数，影响搜索的深度和精确度
    max_iter = st.sidebar.number_input("搜索迭代次数 (最大迭代次数)",
                                 min_value=10,
                                 max_value=2000,
                                 value=100,
                                 help="调整迭代次数来影响算法的搜索深度和精确度。较多的迭代次数可能提高解的质量，但会增加运行时间。")




    #accuracy_percentage = st.sidebar.number_input("输入显示精度 (例如输入 80 代表 ±20%)", value=80,                                                  min_value=50, max_value=100)



    #mu = 130.61737499999998
    #sd = 18.85211

    if st.button('**运行参数拟合**',help="点此按钮进行模型参数拟合"):
        try:
            ssa = SSA(fitness_function, n_dim=n_dim, pop_size=pop_size, max_iter=max_iter, search_space=search_space)
            best_params = ssa.run()
            best_params_values = ssa.gbest_x
            st.session_state['first_part_completed'] = True
            # 将参数和结果显示在 Streamlit 应用上
            best_params_values = [f"{param:.9e}" for param in best_params_values]

            if fix_ea:
                # 固定 E_a 时的参数结果展示
                best_params_dict = {
                    "A": best_params_values[0],
                    "m": best_params_values[1],
                    "p": best_params_values[2],
                    "E_a (固定)": ea_value
                }
            else:
                # 不固定 E_a 时的参数结果展示
                best_params_dict = {
                    "A": best_params_values[0],
                    "m": best_params_values[1],
                    "E_a": best_params_values[2],
                    "p": best_params_values[3]
                }


            best_params_df = pd.DataFrame(best_params_dict, index=[0])
            pd.options.display.float_format = '{:.4f}'.format

            st.markdown("#### 3.2、运行结果")

            st.write("最佳参数：",best_params_df)
            #st.dataframe(best_params_df)
            # 显示最佳损失
            st.write("最佳损失:", ssa.gbest_y)
            # 绘制优化过程图
            st.write("训练损失：")
            fig, ax = plt.subplots()
            ax.plot(ssa.gbest_y_hist)
            ax.set_title('Optimization process')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss')
            st.pyplot(fig)
            # 使用最佳参数计算预测值
            params = ssa.gbest_x
            predicted_beta_1 = function(params, beta_0_1, T_1, VR1, t1)
            predicted_beta_2 = function(params, beta_0_2, T_2, VR2, t2)
            predicted_beta_3 = function(params, beta_0_3, T_3, VR1, t3)
            predicted_beta_4 = function(params, beta_0_4, T_4, VR2, t4)

            # 计算95%置信区间
            z_score = 1.96  # 95%置信区间的Z分数
            #conf_interval_upper = mu + (z_score * sd)
            #conf_interval_lower = mu - (z_score * sd)







            # 用 delta_predicted_beta_1 来计算置信区间
            delta_predicted_beta_1_upper = predicted_beta_1 + (z_score * sd)
            delta_predicted_beta_1_lower = predicted_beta_1 - (z_score * sd)
            delta_predicted_beta_2_upper = predicted_beta_2 + (z_score * sd)
            delta_predicted_beta_2_lower = predicted_beta_2 - (z_score * sd)
            delta_predicted_beta_3_upper = predicted_beta_3 + (z_score * sd)
            delta_predicted_beta_3_lower = predicted_beta_3 - (z_score * sd)
            delta_predicted_beta_4_upper = predicted_beta_4 + (z_score * sd)
            delta_predicted_beta_4_lower = predicted_beta_4 - (z_score * sd)

            # 计算 predicted_beta_1 在 x=0 时的值
            #beta_1_at_0 = function(params, beta_0, T_1, VR1, 0)
            #beta_1_at_1 = function(params, beta_0, T_1, VR1, 0)
            #beta_1_at_2 = function(params, beta_0, T_1, VR1, 0)
            #beta_1_at_3 = function(params, beta_0, T_1, VR1, 0)


            st.write("**拟合结果图像：**")
            # 绘制比较图
            fig1, axs1 = plt.subplots(2, 2, figsize=(10, 8))
            #st.write(beta_1)
            #st.write(beta_1.T)
            #st.write(predicted_beta_150)


            axs1[0,0].plot(t1, beta_1.T, 'o')
            axs1[0,0].plot(t1, predicted_beta_1, 'r-', label='Fitting ')
            axs1[0,0].plot(t1, delta_predicted_beta_1_upper, 'r--', label='95% confidence interval')
            axs1[0,0].plot(t1, delta_predicted_beta_1_lower, 'r--')
            axs1[0,0].set_xlabel('Time')
            axs1[0,0].set_ylabel(ylabel)
            axs1[0,0].set_title('Stress 1')
            axs1[0,0].legend()
            #axs1[0, 0].annotate(f'{beta_1_at_0:.2f}', (0, beta_1_at_0), textcoords="offset points", xytext=(-35, 0),                               )
            #st.write(predicted_beta_1)
            axs1[0,1].plot(t2, beta_2.T, 'o')
            axs1[0,1].plot(t2, predicted_beta_2, 'r-', label='Fitting')
            axs1[0,1].plot(t2, delta_predicted_beta_2_upper, 'r--', label='95% confidence interval')
            axs1[0,1].plot(t2, delta_predicted_beta_2_lower, 'r--')
            axs1[0,1].set_xlabel('Time')
            axs1[0,1].set_ylabel(ylabel)
            axs1[0,1].set_title('Stress 2')
            axs1[0,1].legend()
            #axs1[0, 1].annotate(f'{beta_1_at_0:.2f}', (0, beta_1_at_0), textcoords="offset points", xytext=(-35, 0), )

            axs1[1,0].plot(t3, beta_3.T, 'o')
            axs1[1,0].plot(t3, predicted_beta_3, 'r-', label='Fitting ')
            axs1[1,0].plot(t3, delta_predicted_beta_3_upper, 'r--', label='95% confidence interval')
            axs1[1,0].plot(t3, delta_predicted_beta_3_lower, 'r--')
            axs1[1,0].set_xlabel('Time')
            axs1[1,0].set_ylabel(ylabel)
            axs1[1,0].set_title('Stress 3')
            axs1[1,0].legend()
            #axs1[1, 0].annotate(f'{beta_1_at_0:.2f}', (0, beta_1_at_0), textcoords="offset points", xytext=(-35, 0), )

            axs1[1,1].plot(t4, beta_4.T, 'o')
            axs1[1,1].plot(t4, predicted_beta_4, 'r-', label='Fitting ')
            axs1[1,1].plot(t4, delta_predicted_beta_4_upper, 'r--', label='95% confidence interval')
            axs1[1,1].plot(t4, delta_predicted_beta_4_lower, 'r--')
            axs1[1,1].set_xlabel('Time')
            axs1[1,1].set_ylabel(ylabel)
            axs1[1,1].set_title('Stress 4')
            axs1[1,1].legend()
            #axs1[1, 1].annotate(f'{beta_1_at_0:.2f}', (0, beta_1_at_0), textcoords="offset points", xytext=(-35, 0), )

            axs1[0, 0].set_xlim(left=0)
            axs1[0, 1].set_xlim(left=0)
            axs1[1, 0].set_xlim(left=0)
            axs1[1, 1].set_xlim(left=0)
            # 显示预测值


            plt.tight_layout()
            st.pyplot(fig1)
            st.session_state['params'] = ssa.gbest_x
            #st.session_state['first_part_completed'] = True
            # 创建一个 DataFrame
            predicted_beta_df1 = pd.DataFrame({
                'Stress 1': predicted_beta_1,
            },index=t1)
            predicted_beta_df2 = pd.DataFrame({
                'Stress 2': predicted_beta_2,
            }, index=t2)
            predicted_beta_df3 = pd.DataFrame({
                'Stress 3': predicted_beta_3,
            }, index=t3)
            predicted_beta_df4 = pd.DataFrame({
                'Stress 4': predicted_beta_4,
            }, index=t4)
            # 显示 DataFrame
            st.write("Stress 1 预测结果:", predicted_beta_df1.T)
            st.write("Stress 2 预测结果:", predicted_beta_df2.T)
            st.write("Stress 3 预测结果:", predicted_beta_df3.T)
            st.write("Stress 4 预测结果:", predicted_beta_df4.T)
        except Exception as e:
            st.error(f"运行模型时发生错误：{e}")
##############################
    # =========================
    # 四、元器件可靠度求解
    # =========================
    st.subheader("四、元器件可靠度求解")
    st.markdown("#### 4.1、参数设置")

    if 'params' in st.session_state and st.session_state['params'] is not None:
        params0 = pd.DataFrame({'拟合参数': st.session_state['params']})
        st.write('使用模型参数（A、m、(E_a)、p）:', params0.T)
    else:
        st.warning("尚未进行参数拟合，请先点击 **运行参数拟合**。")
        st.stop()

    st.write('注意：通过 **运行参数拟合** 按钮，可以对模型参数进行更新。')

    o = st.number_input("定义失效阈值：", value=0.95)
    ff = st.number_input("定义图像横坐标显示范围：", value=2000)
    N_mc = st.number_input("Monte Carlo 样本数", min_value=50, max_value=20000, value=500, step=50)

    seed = st.number_input("随机种子（可选，用于复现；-1 表示不固定）", value=-1, step=1)
    root_policy = st.selectbox(
        "寿命取根策略",
        ["legacy_max（与原版一致：倾向取最大根）", "first_crossing_min（推荐：首次达到阈值）"],
        index=0
    )

    st.markdown("#### 4.1.1 可靠度求解应力设置（可覆盖默认 Condition 1~4）")

    use_custom_stress = st.checkbox(
        "使用自定义应力（用于可靠度/寿命分布求解）",
        value=False,
        help="勾选后，可靠度求解将使用这里输入的温度T与电应力V，而不是前面 Condition 1~4 的应力值。"
    )

    if use_custom_stress:
        n_cond = st.number_input("自定义应力条件数量", min_value=1, max_value=12, value=4, step=1)
        T_list, V_list = [], []
        with st.expander("输入每个条件的温度T(K)与电应力V", expanded=True):
            for k in range(int(n_cond)):
                c1, c2 = st.columns(2)
                with c1:
                    Tk = st.number_input(
                        f"Condition {k + 1} 温度 T (K)",
                        value=float([T_1, T_2, T_3, T_4][k] if k < 4 else T_1),
                        format="%.4f",
                        key=f"T_custom_{k}"
                    )
                with c2:
                    Vk = st.number_input(
                        f"Condition {k + 1} 电应力 V",
                        value=float([VR1, VR2, VR3, VR4][k] if k < 4 else VR1),
                        format="%.4f",
                        key=f"V_custom_{k}"
                    )
                T_list.append(Tk)
                V_list.append(Vk)
    else:
        n_cond = 4
        T_list = [T_1, T_2, T_3, T_4]
        V_list = [VR1, VR2, VR3, VR4]

    reliability_clicked = st.button('**可靠度求解**', help="点击此按钮直接求解可靠度")

    if reliability_clicked:
        try:
            params = st.session_state['params']

            # --- Monte Carlo beta0 ---
            if int(seed) >= 0:
                rng = np.random.default_rng(int(seed))
                beta0 = rng.normal(mu, sd, int(N_mc))
            else:
                beta0 = np.random.normal(mu, sd, int(N_mc))

            # 为每个应力条件创建寿命数组
            lifetimes = [np.full(len(beta0), np.nan) for _ in range(int(n_cond))]


            # 失效方程：function(params, ...) - 阈值 = 0
            def equation(t, beta0_i, T, V):
                return function(params, beta0_i, T, V, t) - o


            initial_guesses = [1, 10, 100, 1000]


            # 小工具：对单个样本求寿命
            def solve_lifetime(beta0_i, T_use, V_use):
                sols = []
                for guess in initial_guesses:
                    try:
                        sol = fsolve(equation, guess, args=(beta0_i, T_use, V_use), maxfev=2000)[0]
                        if np.isfinite(sol) and sol > 0:
                            sols.append(float(sol))
                    except Exception:
                        pass

                if len(sols) == 0:
                    return np.nan

                # 去重（避免同一根被多次返回）
                sols = np.array(sorted(sols))
                unique = [sols[0]]
                for s in sols[1:]:
                    if abs(s - unique[-1]) > 1e-6:
                        unique.append(s)
                sols = np.array(unique)

                if root_policy.startswith("legacy_max"):
                    # 尽量复刻原始逻辑：多初值->可能多根；std 很小则取第一个，否则取最大
                    return float(sols[0]) if np.std(sols) <= 1e-5 else float(np.max(sols))
                else:
                    # 推荐：首次达到阈值（最小正根）
                    return float(np.min(sols))


            # --- 求解寿命 ---
            for i in range(len(beta0)):
                for c in range(int(n_cond)):
                    lifetimes[c][i] = solve_lifetime(beta0[i], T_list[c], V_list[c])

            # --- 绘图：R(t), F(t) ---
            import matplotlib.ticker as ticker
            import math

            n = int(n_cond)
            ncols = 2
            nrows = math.ceil(n / ncols)

            fig2, axs2 = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows))  # R(t)
            fig3, axs3 = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows))  # F(t)

            axs2 = np.atleast_2d(axs2)
            axs3 = np.atleast_2d(axs3)

            for idx, lifetime in enumerate(lifetimes):
                row = idx // ncols
                col = idx % ncols

                lifetime = lifetime[np.isfinite(lifetime)]
                lifetime = lifetime[lifetime > 0]  # lognormal 需要正数
                if len(lifetime) < 5:
                    axs2[row, col].set_title(f"Condition {idx + 1} — insufficient samples")
                    axs3[row, col].set_title(f"Condition {idx + 1} — insufficient samples")
                    continue

                shape, loc, scale = stats.lognorm.fit(lifetime, floc=0)

                x = np.linspace(0, ff, 10001)
                F_t = stats.lognorm.cdf(x, shape, loc, scale)
                R_t = 1 - F_t

                axs2[row, col].plot(x, R_t)
                axs2[row, col].set_title(f"Condition {idx + 1} (T={T_list[idx]:.1f}, V={V_list[idx]:.1f}) — R(t)")
                axs2[row, col].set_xlabel("Time(h)")
                axs2[row, col].set_ylabel("R(t)")
                axs2[row, col].yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: f'{val:,.10f}'))

                axs3[row, col].plot(x, F_t)
                axs3[row, col].set_title(f"Condition {idx + 1} (T={T_list[idx]:.1f}, V={V_list[idx]:.1f}) — F(t)")
                axs3[row, col].set_xlabel("Time(h)")
                axs3[row, col].set_ylabel("F(t)")
                axs3[row, col].yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: f'{val:,.10f}'))

                # t_95: R(t)<=0.95 的首次时刻
                t_95_indices = np.where(R_t <= 0.95)[0]
                if len(t_95_indices) > 0:
                    t_95 = x[t_95_indices[0]]
                    axs2[row, col].axvline(x=t_95, color='red', linestyle='--', label=f't(0.95) = {t_95:.2f}')
                    axs2[row, col].axhline(y=0.95, color='red', linestyle='--')
                    axs2[row, col].legend()

                    axs3[row, col].axvline(x=t_95, color='red', linestyle='--', label=f't(F=0.05) = {t_95:.2f}')
                    axs3[row, col].axhline(y=0.05, color='red', linestyle='--')
                    axs3[row, col].legend()

            # 多余子图关掉
            for k in range(n, nrows * ncols):
                r = k // ncols
                c = k % ncols
                axs2[r, c].axis("off")
                axs3[r, c].axis("off")

            plt.tight_layout()
            st.markdown("#### 4.2、可靠度函数图像 R(t)：")
            st.pyplot(fig2)

            plt.tight_layout()
            st.markdown("#### 4.3、累计失效概率图像 F(t) = 1 - R(t)：")
            st.pyplot(fig3)

        except Exception as e:
            st.error(f"处理分布时发生错误：{e}")

    # =======================
    # 五、电路级可靠性求解（静态结构图 + 可选器件类型）
    # =======================
    st.subheader("五、电路级可靠性求解")

    import matplotlib.patches as patches
    import numpy as np
    import matplotlib.pyplot as plt

    # -----------------------
    # 器件类型列表（可以增删）
    # -----------------------
    COMP_TYPES = ["MOSFET", "Diode", "BJT", "IGBT", "OpAmp", "LDO", "Resistor", "Capacitor", "Inductor", "Other"]


    # -----------------------
    # 可靠度组合计算
    # -----------------------
    def clamp01(x):
        try:
            x = float(x)
        except:
            return None
        return max(0.0, min(1.0, x))


    def compute_series(R_list):
        R_list = [clamp01(r) for r in R_list]
        if any(r is None for r in R_list):
            return None
        R = 1.0
        for r in R_list:
            R *= r
        return R


    def compute_parallel(R_list):
        R_list = [clamp01(r) for r in R_list]
        if any(r is None for r in R_list):
            return None
        prod = 1.0
        for r in R_list:
            prod *= (1.0 - r)
        return 1.0 - prod


    # -----------------------
    # 结构图绘制（在方块里显示类型，可选显示R）
    # -----------------------
    def draw_series(n, type_list=None, R_list=None, title="Series Structure", show_R=False):
        fig, ax = plt.subplots(figsize=(8, 2.3))
        ax.set_title(title)
        ax.axis("off")

        x0, y0 = 0.06, 0.45
        w, h = 0.10, 0.28
        gap = 0.05

        # main line
        ax.plot([x0 - 0.04, x0 + n * (w + gap)], [y0 + h / 2, y0 + h / 2], linewidth=2)

        for i in range(n):
            x = x0 + i * (w + gap)
            rect = patches.Rectangle((x, y0), w, h, fill=False, linewidth=2)
            ax.add_patch(rect)

            ctype = type_list[i] if type_list else "C"
            label = f"C{i + 1}\n{ctype}"
            if show_R and R_list is not None:
                label += f"\nR={float(R_list[i]):.3f}"

            ax.text(x + w / 2, y0 + h / 2, label, ha="center", va="center", fontsize=9)

        st.pyplot(fig)


    def draw_parallel(n, type_list=None, R_list=None, title="Parallel Structure", show_R=False):
        fig, ax = plt.subplots(figsize=(8, 3.4))
        ax.set_title(title)
        ax.axis("off")

        xL, xR = 0.12, 0.88
        y_top, y_bot = 0.82, 0.18

        # bus
        ax.plot([xL, xL], [y_bot, y_top], linewidth=2)
        ax.plot([xR, xR], [y_bot, y_top], linewidth=2)

        ys = np.linspace(y_top, y_bot, n)
        w, h = 0.12, 0.14
        for i, y in enumerate(ys):
            ax.plot([xL, 0.40], [y, y], linewidth=2)
            ax.plot([0.60, xR], [y, y], linewidth=2)

            rect = patches.Rectangle((0.40, y - h / 2), w, h, fill=False, linewidth=2)
            ax.add_patch(rect)

            ctype = type_list[i] if type_list else "C"
            label = f"C{i + 1}\n{ctype}"
            if show_R and R_list is not None:
                label += f"\nR={float(R_list[i]):.3f}"

            ax.text(0.40 + w / 2, y, label, ha="center", va="center", fontsize=9)

        st.pyplot(fig)


    def draw_mixed(stages, stage_types, stage_Rs=None, title="Mixed Structure (Series of Stages)", show_R=False):
        """
        stages: list of dict
          {"type":"single"} or {"type":"parallel","k":int}
        stage_types:
          single -> "MOSFET"
          parallel -> ["Diode","MOSFET",...]
        """
        S = len(stages)
        fig_w = max(10, 2.2 * S)
        fig, ax = plt.subplots(figsize=(fig_w, 3.2))
        ax.set_title(title)
        ax.axis("off")

        x = 0.05
        y_mid = 0.55
        stage_w = 0.16
        gap = 0.05

        # main line
        ax.plot([0.02, 0.98], [y_mid, y_mid], linewidth=2)

        for si, stg in enumerate(stages):
            # stage box
            ax.add_patch(patches.Rectangle((x, 0.25), stage_w, 0.60, fill=False, linewidth=1, linestyle="--"))
            ax.text(x + stage_w / 2, 0.87, f"Stage {si + 1}", ha="center", va="center", fontsize=9)

            # optional stage R
            if show_R and stage_Rs is not None and si < len(stage_Rs):
                ax.text(x + stage_w / 2, 0.22, f"R_stage={float(stage_Rs[si]):.3f}", ha="center", va="center",
                        fontsize=8)

            if stg["type"] == "single":
                w, h = 0.07, 0.18
                rect = patches.Rectangle((x + stage_w / 2 - w / 2, y_mid - h / 2), w, h, fill=False, linewidth=2)
                ax.add_patch(rect)

                ctype = stage_types[si]
                ax.text(x + stage_w / 2, y_mid, f"C\n{ctype}", ha="center", va="center", fontsize=9)

            else:
                k = int(stg["k"])
                ys = np.linspace(0.75, 0.35, k)
                # local bus
                xL = x + 0.03
                xR = x + stage_w - 0.03
                ax.plot([xL, xL], [0.35, 0.75], linewidth=2)
                ax.plot([xR, xR], [0.35, 0.75], linewidth=2)

                w, h = 0.06, 0.12
                ctypes = stage_types[si]  # list
                for bi, yy in enumerate(ys):
                    ax.plot([xL, x + stage_w / 2 - 0.03], [yy, yy], linewidth=2)
                    ax.plot([x + stage_w / 2 + 0.03, xR], [yy, yy], linewidth=2)
                    rect = patches.Rectangle((x + stage_w / 2 - w / 2, yy - h / 2), w, h, fill=False, linewidth=2)
                    ax.add_patch(rect)

                    ctype = ctypes[bi] if bi < len(ctypes) else "Other"
                    ax.text(x + stage_w / 2, yy, f"C{bi + 1}\n{ctype}", ha="center", va="center", fontsize=8)

            x += stage_w + gap

        st.pyplot(fig)


    # -----------------------
    # 5.1 串并联模型选择
    # -----------------------
    st.markdown("#### 5.1 串并联模型选择")

    model_type = st.selectbox(
        "选择电路拓扑结构",
        ["串联模型", "并联模型", "混合模型（串联若干级，每级可并联）"]
    )

    t_mission = st.number_input("任务时间/评估时刻（可选，仅用于标注）", value=0.0, min_value=0.0, format="%.4f")
    show_R_on_block = st.checkbox("在结构图中显示每个器件/Stage 的 R 值", value=False)

    # -----------------------
    # 5.2 输入并计算 + 结构图
    # -----------------------
    st.markdown("#### 5.2 电路可靠性求解结果")

    if model_type == "串联模型":
        n = st.number_input("串联元器件数量 n", min_value=1, max_value=200, value=4, step=1)

        st.write("为每个器件选择类型并输入其可靠度 R_i（0~1）：")
        cols = st.columns(3)

        R_list, type_list = [], []
        for i in range(int(n)):
            with cols[i % 3]:
                ctype = st.selectbox(f"C{i + 1} 类型", COMP_TYPES, index=0, key=f"series_type_{i}")
                Ri = st.number_input(f"C{i + 1} 的 R", min_value=0.0, max_value=1.0, value=0.99, format="%.6f",
                                     key=f"series_R_{i}")
            type_list.append(ctype)
            R_list.append(Ri)

        draw_series(int(n), type_list=type_list, R_list=R_list, title="Series Structure", show_R=show_R_on_block)
        R_sys = compute_series(R_list)

    elif model_type == "并联模型":
        n = st.number_input("并联支路元器件数量 n", min_value=1, max_value=50, value=3, step=1)

        st.write("为每个并联支路选择类型并输入其可靠度 R_i（0~1）：")
        cols = st.columns(3)

        R_list, type_list = [], []
        for i in range(int(n)):
            with cols[i % 3]:
                ctype = st.selectbox(f"C{i + 1} 类型", COMP_TYPES, index=1, key=f"par_type_{i}")
                Ri = st.number_input(f"C{i + 1} 的 R", min_value=0.0, max_value=1.0, value=0.95, format="%.6f",
                                     key=f"par_R_{i}")
            type_list.append(ctype)
            R_list.append(Ri)

        draw_parallel(int(n), type_list=type_list, R_list=R_list, title="Parallel Structure", show_R=show_R_on_block)
        R_sys = compute_parallel(R_list)

    else:
        n_stage = st.number_input("Stage 数量（串联级数）", min_value=1, max_value=30, value=3, step=1)

        stages = []
        stage_types = []
        stage_Rs = []

        st.write("为每个 Stage 选择类型，并输入该 Stage 内器件的可靠度：")
        for s in range(int(n_stage)):
            with st.expander(f"Stage {s + 1} 设置", expanded=(s == 0)):
                stg_type = st.selectbox(f"Stage {s + 1} 类型", ["单元件", "并联组"], key=f"mix_stage_type_{s}")

                if stg_type == "单元件":
                    ctype = st.selectbox(f"Stage {s + 1} 器件类型", COMP_TYPES, index=0, key=f"mix_single_type_{s}")
                    Ri = st.number_input(f"Stage {s + 1} 的 R", min_value=0.0, max_value=1.0, value=0.99, format="%.6f",
                                         key=f"mix_single_R_{s}")
                    stages.append({"type": "single"})
                    stage_types.append(ctype)
                    stage_Rs.append(compute_series([Ri]))

                else:
                    k = st.number_input(f"Stage {s + 1} 并联支路数量 k", min_value=2, max_value=50, value=2, step=1,
                                        key=f"mix_par_k_{s}")

                    cols = st.columns(3)
                    R_par, types_par = [], []
                    for j in range(int(k)):
                        with cols[j % 3]:
                            ctype = st.selectbox(f"S{s + 1}-C{j + 1} 类型", COMP_TYPES, index=1,
                                                 key=f"mix_par_type_{s}_{j}")
                            Rij = st.number_input(f"S{s + 1}-C{j + 1} 的 R", min_value=0.0, max_value=1.0, value=0.95,
                                                  format="%.6f", key=f"mix_par_R_{s}_{j}")
                        types_par.append(ctype)
                        R_par.append(Rij)

                    stages.append({"type": "parallel", "k": int(k)})
                    stage_types.append(types_par)
                    stage_Rs.append(compute_parallel(R_par))

        draw_mixed(stages, stage_types, stage_Rs=stage_Rs, title="Mixed Structure: Series of Stages",
                   show_R=show_R_on_block)
        R_sys = compute_series(stage_Rs)


    # -----------------------
    # 输出系统结果
    # -----------------------
    if R_sys is None:
        st.error("电路可靠度计算失败：请检查输入是否为 0~1 的数值。")
    else:
        F_sys = 1.0 - R_sys
        col1, col2, col3 = st.columns(3)
        col1.metric("电路可靠度 R_sys", f"{R_sys:.8f}")
        col2.metric("累计失效概率 F_sys", f"{F_sys:.8f}")
        col3.metric("评估时刻 t", f"{t_mission:.4f}")

        # 可选：把输入和中间结果展开给用户看（混合结构时尤其有用）
        # ✅ 公式用 LaTeX，确保渲染正确
        with st.expander("查看计算明细"):
            st.write("假设：元器件失效相互独立；输入的 $R_i$ 均为同一评估时刻 $t$ 的可靠度。")

            if model_type == "串联模型":
                st.latex(r"R_{\mathrm{sys}}(t)=\prod_{i=1}^{n} R_i(t)")
                st.latex(r"F_{\mathrm{sys}}(t)=1-R_{\mathrm{sys}}(t)")
                st.dataframe(pd.DataFrame({
                    "Component": [f"C{i + 1}" for i in range(len(R_list))],
                    "R_i(t)": R_list
                }))

            elif model_type == "并联模型":
                st.latex(r"R_{\mathrm{sys}}(t)=1-\prod_{i=1}^{n}\left(1-R_i(t)\right)")
                st.latex(r"F_{\mathrm{sys}}(t)=1-R_{\mathrm{sys}}(t)")
                st.dataframe(pd.DataFrame({
                    "Branch": [f"C{i + 1}" for i in range(len(R_list))],
                    "R_i(t)": R_list
                }))

            else:
                st.latex(r"R_{\mathrm{sys}}(t)=\prod_{s=1}^{S} R_{\mathrm{stage},s}(t)")
                st.latex(
                    r"R_{\mathrm{stage}}(t)=1-\prod_{j=1}^{k}\left(1-R_j(t)\right)\quad(\text{if stage is parallel})")
                st.latex(r"F_{\mathrm{sys}}(t)=1-R_{\mathrm{sys}}(t)")
                st.dataframe(pd.DataFrame({
                    "Stage": [f"Stage {i + 1}" for i in range(len(stage_Rs))],
                    "R_stage(t)": stage_Rs
                }))


    #下一步，对于"六、电路级剩余使用寿命求解”，对于这部分，要用机器学习做预测，预测后续，也就是下一部的参数，然后按照我们设置的失效阈值求出剩余使用寿命
    #实现这些指标：
    #► 支持基于物理寿命分析的机器学习模型的训练及验证。
    #● 具备电子产品寿命预估工具，包括：
    #► 寿命预估工具支持在热、电应力条件下的电路寿命预测；
    #► 支持采用物理-数据融合机制的机器学习模型开展寿命预测，至少考虑3 种机器学习模型，如Transformer、LSTM 和RNN 等；
    #► 使用贝叶斯优化、遗传算法或粒子群优化（PSO）等优化算法，针对模型的至少3个参数（如学习率、丢弃率、激活函数等）进行优化；
# =======================
# 六、电路级剩余使用寿命求解（ML + 物理融合）
# =======================
st.subheader("六、电路级剩余使用寿命求解")

# ---------- 依赖检查 ----------
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except Exception as e:
    st.error(
        "缺少 PyTorch 依赖：请先安装 torch。\n"
        "本地：pip install torch\n"
        "Streamlit Cloud：requirements.txt 添加 torch\n\n"
        f"错误信息：{e}"
    )
    st.stop()

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import math
import time

# ---------- 前置检查 ----------
if 'params' not in st.session_state or st.session_state['params'] is None:
    st.warning("尚未进行参数拟合（Section 3）。请先点击 **运行参数拟合**。")
    st.stop()

params_phy = st.session_state['params']

# 如果前面已经输入了阈值 o，就直接用；否则这里补一个
if "o_threshold" not in st.session_state:
    st.session_state["o_threshold"] = 0.95
o_thr = st.number_input("失效阈值 o（用于 RUL 求解）", value=float(st.session_state["o_threshold"]), format="%.6f")
st.session_state["o_threshold"] = o_thr

# 前面已有 K（玻尔兹曼常数 eV/K），若没有就定义
if "K_boltz" not in globals():
    K_boltz = 8.617333262145e-5
else:
    K_boltz = K

# ---------- 选择训练数据类型：正常/增量 ----------
# 前面已经有 formula_type = st.radio(...). 这里做个兜底
formula_type_local = st.radio("训练数据类型", ["增量型数据", "正常数据"], index=1,
                              help="增量型：每个样本减去 t=0 初始值；正常：使用原始绝对值。")

# ---------- 组织四个应力条件数据（来自前面计算出的 DataFrame） ----------
# 要求：每个 condition 的 y_df 为 DataFrame: shape=(n_devices, n_times)
def _safe_df(df_like):
    if isinstance(df_like, pd.DataFrame):
        df = df_like.copy()
    else:
        df = pd.DataFrame(df_like)
    # 强制数值
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(method="ffill", axis=1).fillna(method="bfill", axis=1)
    return df

# 前面有：smoothed1..4 (增量) 和 beta_1_smoothed..4 (正常)
if formula_type_local == "增量型数据":
    y1_df = _safe_df(smoothed1)
    y2_df = _safe_df(smoothed2)
    y3_df = _safe_df(smoothed3)
    y4_df = _safe_df(smoothed4)
    # 增量型：beta0 视为 0（或可用每条曲线的第一点）
    beta0_mode = "zero"
else:
    y1_df = _safe_df(beta_1_smoothed)
    y2_df = _safe_df(beta_2_smoothed)
    y3_df = _safe_df(beta_3_smoothed)
    y4_df = _safe_df(beta_4_smoothed)
    beta0_mode = "first_point"

cond_dict = {
    1: {"t": np.array(t1, dtype=float), "y": y1_df, "T": float(T_1), "V": float(VR1)},
    2: {"t": np.array(t2, dtype=float), "y": y2_df, "T": float(T_2), "V": float(VR2)},
    3: {"t": np.array(t3, dtype=float), "y": y3_df, "T": float(T_3), "V": float(VR3)},
    4: {"t": np.array(t4, dtype=float), "y": y4_df, "T": float(T_4), "V": float(VR4)},
}

# ---------- Physics-Data Fusion 方式 ----------
fusion_mode = st.selectbox(
    "物理-数据融合方式（Physics–Data Fusion）",
    [
        "纯数据驱动（Data-only）",
        "物理作为额外特征（Physics as Feature）",
        "残差学习（y = y_phys + Δy_ml）",
        "物理正则（Loss += λ·MSE(y_pred, y_phys)）",
    ],
    index=2
)
lambda_phy = 0.0
if fusion_mode == "物理正则（Loss += λ·MSE(y_pred, y_phys)）":
    lambda_phy = st.number_input("物理正则系数 λ", min_value=0.0, max_value=10.0, value=0.3, step=0.1)

# ---------- 模型选择 ----------
model_type = st.selectbox("选择预测模型", ["RNN", "LSTM", "Transformer"], index=1)

# ---------- 训练设置 ----------
st.markdown("#### 6.1 训练与验证设置")
lookback = st.number_input("输入序列长度 lookback", min_value=3, max_value=200, value=10, step=1)
horizon = st.number_input("预测步长 horizon（一次预测未来点数）", min_value=1, max_value=200, value=5, step=1)

batch_size = st.number_input("batch size", min_value=8, max_value=2048, value=64, step=8)
epochs = st.number_input("epochs", min_value=1, max_value=500, value=50, step=1)
val_ratio = st.number_input("验证集比例", min_value=0.05, max_value=0.5, value=0.2, step=0.05)

# ---------- 统一激活函数 ----------
ACTS = ["relu", "tanh", "gelu"]
def get_activation(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    return nn.ReLU()

# ---------- 构造 ML 数据集 ----------
def compute_y_phys(t_arr, beta0, T, V):
    # 直接调用前面的物理模型 function(params, beta0, T, V, t)
    # 注意：function 里 V 使用 abs(V) 已经做了；如果没做，请自行 abs
    return np.array(function(params_phy, beta0, T, V, t_arr), dtype=float)

def make_samples_from_condition(cond_id, lookback, horizon):
    info = cond_dict[cond_id]
    t = info["t"]
    y_df = info["y"]
    T = info["T"]
    V = info["V"]

    # 统一长度：t 的长度应等于列数
    L = min(len(t), y_df.shape[1])
    t = t[:L]
    y_df = y_df.iloc[:, :L].copy()

    X_list, Y_list, PHYF_list = [], [], []
    # 每一行是一条“器件/样本”的时间序列
    for r in range(y_df.shape[0]):
        y = y_df.iloc[r, :].to_numpy(dtype=float)
        # beta0 规则
        if beta0_mode == "zero":
            beta0 = 0.0
        else:
            beta0 = float(y[0])

        y_phys = compute_y_phys(t, beta0, T, V)

        # 滑窗采样
        max_start = L - lookback - horizon
        if max_start < 0:
            continue
        for s in range(max_start + 1):
            y_hist = y[s:s+lookback]
            y_fut  = y[s+lookback:s+lookback+horizon]

            t_hist = t[s:s+lookback]
            t_fut  = t[s+lookback:s+lookback+horizon]

            y_phys_hist = y_phys[s:s+lookback]
            y_phys_fut  = y_phys[s+lookback:s+lookback+horizon]

            # 特征拼装
            # 基本：y_hist
            feats = [y_hist.reshape(-1, 1)]

            # 加入应力特征（每个时间步重复一次）
            T_feat = np.full((lookback, 1), T, dtype=float)
            V_feat = np.full((lookback, 1), V, dtype=float)

            feats.append(T_feat)
            feats.append(V_feat)

            # 可选：归一化时间（提高泛化）
            t_norm = (t_hist - t_hist.min()) / (t_hist.max() - t_hist.min() + 1e-12)
            feats.append(t_norm.reshape(-1, 1))

            if fusion_mode == "物理作为额外特征（Physics as Feature）":
                feats.append(y_phys_hist.reshape(-1, 1))
                X = np.concatenate(feats, axis=1)
                Y = y_fut
                PHYF = y_phys_fut
            elif fusion_mode == "残差学习（y = y_phys + Δy_ml）":
                # 输入用残差序列，输出预测残差
                res_hist = (y_hist - y_phys_hist).reshape(-1, 1)
                feats[0] = res_hist  # 替换第一路
                X = np.concatenate(feats, axis=1)
                Y = (y_fut - y_phys_fut)  # 预测 Δy
                PHYF = y_phys_fut         # 后面加回
            else:
                # Data-only 或 Physics-regularized 都用 y_hist 直接预测 y_fut
                X = np.concatenate(feats, axis=1)
                Y = y_fut
                PHYF = y_phys_fut

            X_list.append(X)
            Y_list.append(Y)
            PHYF_list.append(PHYF)

    if len(X_list) == 0:
        return None

    X = np.stack(X_list, axis=0)               # (N, lookback, n_features)
    Y = np.stack(Y_list, axis=0)               # (N, horizon)
    PHYF = np.stack(PHYF_list, axis=0)         # (N, horizon)
    return X, Y, PHYF

def build_dataset(lookback, horizon, cond_ids=(1,2,3,4)):
    X_all, Y_all, P_all = [], [], []
    for cid in cond_ids:
        out = make_samples_from_condition(cid, lookback, horizon)
        if out is None:
            continue
        X, Y, P = out
        X_all.append(X); Y_all.append(Y); P_all.append(P)
    if len(X_all) == 0:
        return None
    X_all = np.concatenate(X_all, axis=0)
    Y_all = np.concatenate(Y_all, axis=0)
    P_all = np.concatenate(P_all, axis=0)
    return X_all, Y_all, P_all

class SeqDataset(Dataset):
    def __init__(self, X, Y, P):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.P = torch.tensor(P, dtype=torch.float32)  # physics future (for fusion / reg)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.P[idx]

# ---------- 模型定义 ----------
class RNNForecaster(nn.Module):
    def __init__(self, n_features, hidden_dim, n_layers, dropout, act_name, horizon):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0
        )
        self.act = get_activation(act_name)
        self.head = nn.Linear(hidden_dim, horizon)
    def forward(self, x):
        out, _ = self.rnn(x)               # (B, T, H)
        h_last = out[:, -1, :]             # (B, H)
        h_last = self.act(h_last)
        return self.head(h_last)           # (B, horizon)

class LSTMForecaster(nn.Module):
    def __init__(self, n_features, hidden_dim, n_layers, dropout, act_name, horizon):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0
        )
        self.act = get_activation(act_name)
        self.head = nn.Linear(hidden_dim, horizon)
    def forward(self, x):
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]
        h_last = self.act(h_last)
        return self.head(h_last)

class TransformerForecaster(nn.Module):
    def __init__(self, n_features, d_model, nhead, num_layers, dropout, act_name, horizon):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True, activation=act_name
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, horizon)
    def forward(self, x):
        z = self.proj(x)                   # (B, T, d_model)
        z = self.encoder(z)                # (B, T, d_model)
        z_last = z[:, -1, :]
        return self.head(z_last)

def build_model(model_type, n_features, hp, horizon):
    if model_type == "RNN":
        return RNNForecaster(
            n_features=n_features,
            hidden_dim=int(hp["hidden_dim"]),
            n_layers=int(hp["n_layers"]),
            dropout=float(hp["dropout"]),
            act_name=hp["act"],
            horizon=horizon
        )
    if model_type == "LSTM":
        return LSTMForecaster(
            n_features=n_features,
            hidden_dim=int(hp["hidden_dim"]),
            n_layers=int(hp["n_layers"]),
            dropout=float(hp["dropout"]),
            act_name=hp["act"],
            horizon=horizon
        )
    # Transformer
    return TransformerForecaster(
        n_features=n_features,
        d_model=int(hp["d_model"]),
        nhead=int(hp["nhead"]),
        num_layers=int(hp["n_layers"]),
        dropout=float(hp["dropout"]),
        act_name=hp["act"],
        horizon=horizon
    )

# ---------- 训练/验证 ----------
def train_one(model, train_loader, val_loader, lr, epochs, lambda_phy, device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    patience = 10
    bad = 0

    for ep in range(int(epochs)):
        model.train()
        for xb, yb, pb in train_loader:
            xb = xb.to(device); yb = yb.to(device); pb = pb.to(device)

            pred = model(xb)

            # 残差学习：pred 是 Δy，最终 y_hat = y_phys + Δy
            if fusion_mode == "残差学习（y = y_phys + Δy_ml）":
                y_hat = pb + pred
                loss_main = mse(y_hat, pb + yb)  # yb 是 Δy_true
            else:
                y_hat = pred
                loss_main = mse(y_hat, yb)

            # 物理正则：让预测靠近 y_phys（注意：对残差学习时同样以 y_hat 为准）
            if fusion_mode == "物理正则（Loss += λ·MSE(y_pred, y_phys)）" and lambda_phy > 0:
                loss_phy = mse(y_hat, pb)
                loss = loss_main + lambda_phy * loss_phy
            else:
                loss = loss_main

            opt.zero_grad()
            loss.backward()
            opt.step()

        # val
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb, pb in val_loader:
                xb = xb.to(device); yb = yb.to(device); pb = pb.to(device)
                pred = model(xb)
                if fusion_mode == "残差学习（y = y_phys + Δy_ml）":
                    y_hat = pb + pred
                    y_true = pb + yb
                else:
                    y_hat = pred
                    y_true = yb
                preds.append(y_hat.cpu().numpy())
                trues.append(y_true.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        val_rmse = math.sqrt(mean_squared_error(trues.flatten(), preds.flatten()))

        if val_rmse < best_val - 1e-6:
            best_val = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val

def evaluate(model, loader, device="cpu"):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb, pb in loader:
            xb = xb.to(device); yb = yb.to(device); pb = pb.to(device)
            pred = model(xb)
            if fusion_mode == "残差学习（y = y_phys + Δy_ml）":
                y_hat = pb + pred
                y_true = pb + yb
            else:
                y_hat = pred
                y_true = yb
            preds.append(y_hat.cpu().numpy())
            trues.append(y_true.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    rmse = math.sqrt(mean_squared_error(trues.flatten(), preds.flatten()))
    mae = mean_absolute_error(trues.flatten(), preds.flatten())
    return rmse, mae

# ---------- 超参数空间 ----------
def default_hp_space(model_type):
    if model_type in ["RNN", "LSTM"]:
        return {
            "lr": (1e-4, 5e-3),
            "hidden_dim": (16, 256),
            "n_layers": (1, 3),
            "dropout": (0.0, 0.5),
            "act": ["relu", "tanh", "gelu"],
        }
    else:
        return {
            "lr": (1e-4, 5e-3),
            "d_model": (32, 256),
            "nhead": (2, 8),
            "n_layers": (1, 3),
            "dropout": (0.0, 0.5),
            "act": ["relu", "gelu"],  # Transformer 常用 relu/gelu
        }

# ---------- 优化算法：Bayes / GA / PSO ----------
opt_method = st.selectbox("超参数优化算法", ["不优化（手动）", "贝叶斯优化", "遗传算法 GA", "粒子群 PSO"], index=0)

# 手动超参数（若不优化）
st.markdown("#### 6.2 模型超参数（至少 3 个）")
if model_type in ["RNN", "LSTM"]:
    hp_manual = {
        "lr": st.number_input("学习率 lr", min_value=1e-5, max_value=1e-1, value=1e-3, format="%.6f"),
        "hidden_dim": st.number_input("hidden_dim", min_value=8, max_value=512, value=64, step=8),
        "n_layers": st.number_input("层数 n_layers", min_value=1, max_value=6, value=2, step=1),
        "dropout": st.number_input("dropout", min_value=0.0, max_value=0.9, value=0.2, step=0.05),
        "act": st.selectbox("激活函数 act", ACTS, index=2),
    }
else:
    hp_manual = {
        "lr": st.number_input("学习率 lr", min_value=1e-5, max_value=1e-1, value=1e-3, format="%.6f"),
        "d_model": st.number_input("d_model", min_value=16, max_value=512, value=64, step=8),
        "nhead": st.number_input("nhead", min_value=1, max_value=16, value=4, step=1),
        "n_layers": st.number_input("层数 n_layers", min_value=1, max_value=6, value=2, step=1),
        "dropout": st.number_input("dropout", min_value=0.0, max_value=0.9, value=0.2, step=0.05),
        "act": st.selectbox("激活函数 act", ["relu", "gelu"], index=1),
    }

opt_trials = st.number_input("优化评估次数/迭代数", min_value=5, max_value=200, value=20, step=1)

def sample_from_space(space):
    hp = {}
    for k, v in space.items():
        if isinstance(v, tuple):
            lo, hi = v
            if k in ["hidden_dim", "d_model", "n_layers", "nhead"]:
                hp[k] = int(np.random.randint(int(lo), int(hi) + 1))
            else:
                hp[k] = float(np.random.uniform(lo, hi))
        else:
            hp[k] = str(np.random.choice(v))
    # 合法性修正（Transformer nhead 必须整除 d_model）
    if model_type == "Transformer":
        d = int(hp["d_model"])
        h = int(hp["nhead"])
        # 调整 nhead 使 d_model % nhead == 0
        candidates = [x for x in range(1, 17) if d % x == 0]
        hp["nhead"] = int(min(candidates, key=lambda x: abs(x - h))) if candidates else 1
    return hp

# GA / PSO 用连续向量编码的简化实现
def hp_to_vec(hp, space):
    vec = []
    keys = []
    for k, v in space.items():
        if isinstance(v, tuple):
            lo, hi = v
            vec.append((float(hp[k]) - lo) / (hi - lo + 1e-12))
            keys.append(k)
        else:
            # categorical：映射到 [0,1]
            idx = v.index(hp[k])
            vec.append(idx / max(1, len(v) - 1))
            keys.append(k)
    return np.array(vec, dtype=float), keys

def vec_to_hp(vec, keys, space):
    hp = {}
    for i, k in enumerate(keys):
        v = space[k]
        if isinstance(v, tuple):
            lo, hi = v
            x = lo + float(vec[i]) * (hi - lo)
            if k in ["hidden_dim", "d_model", "n_layers", "nhead"]:
                hp[k] = int(np.clip(round(x), lo, hi))
            else:
                hp[k] = float(np.clip(x, lo, hi))
        else:
            idx = int(round(float(vec[i]) * (len(v) - 1)))
            idx = int(np.clip(idx, 0, len(v) - 1))
            hp[k] = str(v[idx])
    if model_type == "Transformer":
        d = int(hp["d_model"])
        h = int(hp["nhead"])
        candidates = [x for x in range(1, 17) if d % x == 0]
        hp["nhead"] = int(min(candidates, key=lambda x: abs(x - h))) if candidates else 1
    return hp

# ---------- 训练数据准备 ----------
with st.expander("查看训练数据规模", expanded=False):
    data_pack = build_dataset(int(lookback), int(horizon), cond_ids=(1,2,3,4))
    if data_pack is None:
        st.error("数据不足：无法构建训练样本。请检查 t 序列长度、lookback/horizon、以及数据表行列范围。")
    else:
        X_all, Y_all, P_all = data_pack
        st.write("X shape:", X_all.shape, "Y shape:", Y_all.shape, "Physics shape:", P_all.shape)

# ---------- 目标函数：给优化算法用 ----------
def quick_objective(hp, X_all, Y_all, P_all):
    # 少量 epoch 用于评估
    X_tr, X_va, Y_tr, Y_va, P_tr, P_va = train_test_split(
        X_all, Y_all, P_all, test_size=float(val_ratio), random_state=42
    )
    ds_tr = SeqDataset(X_tr, Y_tr, P_tr)
    ds_va = SeqDataset(X_va, Y_va, P_va)
    dl_tr = DataLoader(ds_tr, batch_size=int(batch_size), shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=int(batch_size), shuffle=False)

    n_features = X_all.shape[2]
    model = build_model(model_type, n_features, hp, int(horizon))
    model, val_rmse = train_one(model, dl_tr, dl_va, lr=float(hp["lr"]),
                                epochs=min(15, int(epochs)), lambda_phy=float(lambda_phy), device="cpu")
    return val_rmse

# ---------- 三种优化实现 ----------
def run_bayes_opt(space, X_all, Y_all, P_all):
    from bayes_opt import BayesianOptimization

    # bayes_opt 只支持连续变量：对 categorical act 用索引
    cat_keys = [k for k, v in space.items() if not isinstance(v, tuple)]
    cont_keys = [k for k, v in space.items() if isinstance(v, tuple)]

    pbounds = {}
    for k in cont_keys:
        lo, hi = space[k]
        pbounds[k] = (float(lo), float(hi))
    for k in cat_keys:
        pbounds[k] = (0.0, float(len(space[k]) - 1))

    def f_target(**kwargs):
        hp = {}
        for k in cont_keys:
            if k in ["hidden_dim", "d_model", "n_layers", "nhead"]:
                hp[k] = int(round(kwargs[k]))
            else:
                hp[k] = float(kwargs[k])
        for k in cat_keys:
            idx = int(round(kwargs[k]))
            idx = max(0, min(idx, len(space[k]) - 1))
            hp[k] = space[k][idx]

        # 修正 Transformer nhead
        if model_type == "Transformer":
            d = int(hp["d_model"])
            h = int(hp["nhead"])
            candidates = [x for x in range(1, 17) if d % x == 0]
            hp["nhead"] = int(min(candidates, key=lambda x: abs(x - h))) if candidates else 1

        rmse = quick_objective(hp, X_all, Y_all, P_all)
        return -rmse  # maximize

    bo = BayesianOptimization(f=f_target, pbounds=pbounds, verbose=0, random_state=7)
    bo.maximize(init_points=5, n_iter=max(1, int(opt_trials) - 5))

    best = bo.max["params"]
    hp_best = {}
    for k in cont_keys:
        if k in ["hidden_dim", "d_model", "n_layers", "nhead"]:
            hp_best[k] = int(round(best[k]))
        else:
            hp_best[k] = float(best[k])
    for k in cat_keys:
        idx = int(round(best[k]))
        idx = max(0, min(idx, len(space[k]) - 1))
        hp_best[k] = space[k][idx]

    if model_type == "Transformer":
        d = int(hp_best["d_model"])
        h = int(hp_best["nhead"])
        candidates = [x for x in range(1, 17) if d % x == 0]
        hp_best["nhead"] = int(min(candidates, key=lambda x: abs(x - h))) if candidates else 1

    return hp_best

def run_ga(space, X_all, Y_all, P_all, pop=12, gens=10, mut=0.15):
    # 简化 GA：在向量空间做选择/交叉/变异
    # 迭代次数用 opt_trials 映射
    gens = max(3, int(opt_trials) // 2)

    # 初始化
    init = [sample_from_space(space) for _ in range(pop)]
    vecs, keys = zip(*[hp_to_vec(h, space) for h in init])
    vecs = np.stack(vecs, axis=0)
    keys = keys[0]

    def fitness(v):
        hp = vec_to_hp(np.clip(v, 0, 1), keys, space)
        rmse = quick_objective(hp, X_all, Y_all, P_all)
        return rmse

    best_hp, best_rmse = None, float("inf")

    for g in range(gens):
        rmses = np.array([fitness(vecs[i]) for i in range(pop)], dtype=float)
        order = np.argsort(rmses)
        vecs = vecs[order]
        rmses = rmses[order]

        if rmses[0] < best_rmse:
            best_rmse = float(rmses[0])
            best_hp = vec_to_hp(vecs[0], keys, space)

        # 选择前半作为父代
        parents = vecs[: pop // 2]

        # 交叉生成子代
        children = []
        while len(children) < pop - len(parents):
            a, b = parents[np.random.randint(len(parents))], parents[np.random.randint(len(parents))]
            mask = np.random.rand(len(keys)) < 0.5
            child = np.where(mask, a, b)
            # 变异
            m = np.random.rand(len(keys)) < mut
            child[m] = child[m] + 0.1 * np.random.randn(np.sum(m))
            children.append(np.clip(child, 0, 1))

        vecs = np.vstack([parents, np.stack(children, axis=0)])

    return best_hp

def run_pso(space, X_all, Y_all, P_all, swarm=12, iters=15):
    iters = max(5, int(opt_trials))
    # 简化 PSO：在 [0,1]^d 空间
    # 初始化粒子
    hp0 = [sample_from_space(space) for _ in range(swarm)]
    vecs, keys = zip(*[hp_to_vec(h, space) for h in hp0])
    X = np.stack(vecs, axis=0)
    keys = keys[0]
    V = 0.1 * np.random.randn(*X.shape)

    def score(v):
        hp = vec_to_hp(np.clip(v, 0, 1), keys, space)
        return quick_objective(hp, X_all, Y_all, P_all)

    pbest = X.copy()
    pbest_score = np.array([score(pbest[i]) for i in range(swarm)], dtype=float)
    gbest_idx = int(np.argmin(pbest_score))
    gbest = pbest[gbest_idx].copy()
    gbest_score = float(pbest_score[gbest_idx])

    w, c1, c2 = 0.6, 1.2, 1.2

    for it in range(iters):
        r1 = np.random.rand(*X.shape)
        r2 = np.random.rand(*X.shape)
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        X = np.clip(X + V, 0, 1)

        scores = np.array([score(X[i]) for i in range(swarm)], dtype=float)
        improve = scores < pbest_score
        pbest[improve] = X[improve]
        pbest_score[improve] = scores[improve]

        gidx = int(np.argmin(pbest_score))
        if float(pbest_score[gidx]) < gbest_score:
            gbest_score = float(pbest_score[gidx])
            gbest = pbest[gidx].copy()

    return vec_to_hp(gbest, keys, space)

# ---------- 训练按钮 ----------
train_clicked = st.button("🚀 训练/验证（Section 6）", help="训练所选模型，并给出验证指标；可选先做超参数优化")

if train_clicked:
    data_pack = build_dataset(int(lookback), int(horizon), cond_ids=(1,2,3,4))
    if data_pack is None:
        st.error("数据不足：无法训练。请检查 lookback/horizon 与数据长度。")
        st.stop()

    X_all, Y_all, P_all = data_pack
    X_tr, X_va, Y_tr, Y_va, P_tr, P_va = train_test_split(
        X_all, Y_all, P_all, test_size=float(val_ratio), random_state=42
    )

    ds_tr = SeqDataset(X_tr, Y_tr, P_tr)
    ds_va = SeqDataset(X_va, Y_va, P_va)
    dl_tr = DataLoader(ds_tr, batch_size=int(batch_size), shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=int(batch_size), shuffle=False)

    space = default_hp_space(model_type)

    # 选择超参数
    if opt_method == "不优化（手动）":
        hp_best = hp_manual
    elif opt_method == "贝叶斯优化":
        with st.spinner("Bayesian Optimization 进行中..."):
            hp_best = run_bayes_opt(space, X_all, Y_all, P_all)
    elif opt_method == "遗传算法 GA":
        with st.spinner("GA 进行中..."):
            hp_best = run_ga(space, X_all, Y_all, P_all, pop=12, gens=max(6, int(opt_trials)//2), mut=0.2)
    else:
        with st.spinner("PSO 进行中..."):
            hp_best = run_pso(space, X_all, Y_all, P_all, swarm=12, iters=int(opt_trials))

    st.success("超参数确定完成 ✅")
    st.write("Best HP:", hp_best)

    n_features = X_all.shape[2]
    model = build_model(model_type, n_features, hp_best, int(horizon))

    with st.spinner("训练最终模型..."):
        model, best_val = train_one(model, dl_tr, dl_va, lr=float(hp_best["lr"]),
                                    epochs=int(epochs), lambda_phy=float(lambda_phy), device="cpu")

    rmse, mae = evaluate(model, dl_va, device="cpu")
    st.write(f"验证集 RMSE: {rmse:.6f}")
    st.write(f"验证集 MAE : {mae:.6f}")

    # 保存到 session_state 供后续 RUL 预测使用
    st.session_state["rul_model"] = model
    st.session_state["rul_hp"] = hp_best
    st.session_state["rul_cfg"] = {
        "lookback": int(lookback),
        "horizon": int(horizon),
        "fusion_mode": fusion_mode,
        "beta0_mode": beta0_mode,
        "model_type": model_type,
        "lambda_phy": float(lambda_phy),
        "formula_type": formula_type_local,
    }

# =======================
# 6.3 基于预测序列 + 阈值 求 RUL + 绘图（完整、可复用、不会越改越乱）
# =======================
st.markdown("#### 6.3 电路级 RUL 预测（先预测退化，再按阈值求寿命）")

if "rul_model" not in st.session_state or st.session_state["rul_model"] is None:
    st.info("请先在上面点击 **训练/验证**，得到模型后再做 RUL 预测。")
else:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import math

    # torch 兜底（上面 Section 6 已 import，这里避免作用域问题）
    try:
        import torch
    except Exception as e:
        st.error(f"torch 未正确导入：{e}")
        st.stop()

    # -----------------------
    # 读取训练得到的模型与配置（保证预测与训练一致）
    # -----------------------
    model = st.session_state["rul_model"]
    cfg = st.session_state.get("rul_cfg", {})
    lookback = int(cfg.get("lookback", 5))
    horizon  = int(cfg.get("horizon", 2))
    fusion_mode_cfg = cfg.get("fusion_mode", "残差学习（y = y_phys + Δy_ml）")
    beta0_mode_cfg  = cfg.get("beta0_mode", "zero")

    # -----------------------
    # 选择电路拓扑 + 工况
    # -----------------------
    topo = st.selectbox(
        "电路拓扑结构（用于电路级 RUL 聚合）",
        ["串联模型", "并联模型", "混合模型（串联若干级，每级可并联）"],
        index=0,
        key="sec63_topo"
    )

    cond_sel = st.selectbox("预测工况（选择应力条件）", [1, 2, 3, 4], index=0, key="sec63_cond")
    info = cond_dict[int(cond_sel)]
    t_axis = np.array(info["t"], dtype=float)
    y_df   = info["y"]
    T_use  = float(info["T"])
    V_use  = float(info["V"])

    # 当前时刻索引
    cur_idx = st.number_input(
        "当前时刻索引（从 0 开始）",
        min_value=int(lookback),
        max_value=int(len(t_axis) - 2),
        value=int(len(t_axis) - 2),
        step=1,
        help="RUL 从该时刻开始计算；需要保证前面有 lookback 个点。",
        key="sec63_cur_idx"
    )

    # 阈值触发方向（非常关键）
    fail_rule = st.selectbox(
        "失效判据（阈值触发方向）",
        ["退化量上升到阈值（y >= o）", "健康度下降到阈值（y <= o）"],
        index=0,
        key="sec63_fail_rule"
    )

    # 外推参数
    max_roll = st.number_input("最大滚动次数 max_roll", min_value=10, max_value=5000, value=10, step=50, key="sec63_max_roll")
    dt_default = float(np.median(np.diff(t_axis))) if len(t_axis) > 2 else 1.0
    dt_user = st.number_input("未来时间步长 dt", value=float(dt_default), format="%.6f", key="sec63_dt")

    # 数据范围提示（判断阈值是否合理）
    y_arr_all = y_df.to_numpy(dtype=float)
    y_min = float(np.nanmin(y_arr_all))
    y_max = float(np.nanmax(y_arr_all))
    st.caption(f"当前工况数据范围：y ∈ [{y_min:.6g}, {y_max:.6g}]；当前阈值 o = {float(o_thr):.6g}")

    # -----------------------
    # 小工具：清洗单条曲线，避免 NaN/Inf 导致预测失败
    # -----------------------
    def _clean_curve(y_curve: np.ndarray) -> np.ndarray:
        y = np.array(y_curve, dtype=float)
        y = np.where(np.isfinite(y), y, np.nan)
        # 先前向/后向填充，再用 0 兜底（极端情况）
        if np.all(np.isnan(y)):
            return np.zeros_like(y, dtype=float)
        # forward fill
        for i in range(1, len(y)):
            if np.isnan(y[i]):
                y[i] = y[i-1]
        # backward fill
        for i in range(len(y)-2, -1, -1):
            if np.isnan(y[i]):
                y[i] = y[i+1]
        # still nan -> 0
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return y

    # -----------------------
    # UI：选择电路元器件（行索引映射）
    # -----------------------
    st.markdown("##### 6.3.1 选择电路中的元器件（从数据表行中映射）")

    def pick_components_ui(prefix, n):
        comps = []
        cols = st.columns(3)
        for i in range(int(n)):
            with cols[i % 3]:
                ctype = st.selectbox(
                    f"{prefix}C{i+1} 类型",
                    ["MOSFET", "Diode", "BJT", "IGBT", "Other"],
                    index=0,
                    key=f"{prefix}_type_{i}"
                )
                ridx = st.number_input(
                    f"{prefix}C{i+1} 数据行索引",
                    min_value=0,
                    max_value=int(y_df.shape[0] - 1),
                    value=min(i, int(y_df.shape[0] - 1)),
                    step=1,
                    key=f"{prefix}_row_{i}"
                )
            comps.append({"name": f"C{i+1}", "type": ctype, "row": int(ridx)})
        return comps

    comps = None
    stages_spec = None  # 混合结构用
    if topo == "串联模型":
        n = st.number_input("串联元器件数量 n（用于本次 RUL 聚合）", min_value=1, max_value=50, value=4, step=1, key="sec63_n_series")
        comps = pick_components_ui("sec63_series_", int(n))

    elif topo == "并联模型":
        n = st.number_input("并联支路数量 n（用于本次 RUL 聚合）", min_value=1, max_value=50, value=3, step=1, key="sec63_n_par")
        comps = pick_components_ui("sec63_par_", int(n))

    else:
        # 混合结构：Stage 串联，每个 Stage 可并联
        S = st.number_input("Stage 数量（串联级数）", min_value=1, max_value=20, value=3, step=1, key="sec63_stage_S")
        stages_spec = []
        all_comps = []

        st.write("为每个 Stage 设置类型，并选择该 Stage 内元器件行索引：")
        for s in range(int(S)):
            with st.expander(f"Stage {s+1} 设置", expanded=(s == 0)):
                stg_type = st.selectbox(f"Stage {s+1} 类型", ["单元件", "并联组"], key=f"sec63_stage_type_{s}")
                if stg_type == "单元件":
                    c = pick_components_ui(f"sec63_mix_s{s+1}_", 1)[0]
                    c["name"] = f"S{s+1}_C1"
                    all_comps.append(c)
                    stages_spec.append({"stage": s+1, "type": "single", "comps": [c]})
                else:
                    k = st.number_input(f"Stage {s+1} 并联支路数量 k", min_value=2, max_value=20, value=2, step=1, key=f"sec63_stage_k_{s}")
                    cs = pick_components_ui(f"sec63_mix_s{s+1}_", int(k))
                    for j, c in enumerate(cs, start=1):
                        c["name"] = f"S{s+1}_C{j}"
                        all_comps.append(c)
                    stages_spec.append({"stage": s+1, "type": "parallel", "comps": cs})

        comps = all_comps  # 用于统一计算与绘图

    # -----------------------
    # 绘图函数：测量 + 预测 + 阈值 + t_fail
    # -----------------------
    def plot_rul_prediction(
        t_axis, y_curve,
        cur_idx, lookback,
        pred_times, pred_vals,
        threshold,
        t_fail=None, RUL=None,
        title=""
    ):
        t_axis = np.array(t_axis, dtype=float)
        y_curve = np.array(y_curve, dtype=float)

        fig, ax = plt.subplots(figsize=(9, 4))

        ax.plot(t_axis, y_curve, "o-", alpha=0.6, label="Measured")

        lb_start = max(0, int(cur_idx) - int(lookback))
        ax.plot(
            t_axis[lb_start:int(cur_idx)],
            y_curve[lb_start:int(cur_idx)],
            "o-",
            linewidth=2.5,
            label="History window (lookback)"
        )

        t_cur = float(t_axis[int(cur_idx)])
        ax.axvline(t_cur, linestyle="--", linewidth=2, label=f"t_current={t_cur:.3g}")

        if pred_times is not None and len(pred_times) > 0:
            ax.plot(pred_times, pred_vals, "-", linewidth=2, label="Predicted (rollout)")

        ax.axhline(float(threshold), linestyle="--", linewidth=2, label=f"Threshold={float(threshold):.3g}")

        if t_fail is not None and np.isfinite(t_fail):
            ax.axvline(float(t_fail), color="red", linestyle="--", linewidth=2, label=f"t_fail={float(t_fail):.3g}")
            if RUL is not None and np.isfinite(RUL):
                ax.text(float(t_fail), float(threshold), f"  RUL={float(RUL):.3g}", color="red", va="bottom")

        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("y (degradation / health indicator)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

        if pred_times is not None and len(pred_times) > 0:
            ax.set_xlim(min(t_axis.min(), np.min(pred_times)), max(t_axis.max(), np.max(pred_times)))
        else:
            ax.set_xlim(t_axis.min(), t_axis.max())

        return fig

    # -----------------------
    # 单器件：构造模型输入（必须与训练时一致）
    # -----------------------
    def build_x_input(y_hist, t_hist, T, V, y_phys_hist=None):
        feats = []
        if fusion_mode_cfg == "残差学习（y = y_phys + Δy_ml）" and y_phys_hist is not None:
            feats.append((y_hist - y_phys_hist).reshape(-1, 1))
        else:
            feats.append(y_hist.reshape(-1, 1))

        feats.append(np.full((len(y_hist), 1), T, dtype=float))
        feats.append(np.full((len(y_hist), 1), V, dtype=float))

        t_norm = (t_hist - t_hist.min()) / (t_hist.max() - t_hist.min() + 1e-12)
        feats.append(t_norm.reshape(-1, 1))

        if fusion_mode_cfg == "物理作为额外特征（Physics as Feature）" and y_phys_hist is not None:
            feats.append(y_phys_hist.reshape(-1, 1))

        X = np.concatenate(feats, axis=1)
        return torch.tensor(X[None, ...], dtype=torch.float32)

    # -----------------------
    # 单器件：滚动预测直到触阈，得到 RUL
    # 依赖：compute_y_phys(t_arr, beta0, T, V) 需已在 Section 6 前定义
    # -----------------------
    def predict_rul_for_one_curve(y_curve, t_axis, T, V, cur_idx, threshold, max_roll=400, dt=1.0):
        """
        返回：RUL, t_fail, traj, status
        traj = (pred_times, pred_vals)
        """
        y_curve = _clean_curve(np.array(y_curve, dtype=float))
        t_axis = np.array(t_axis, dtype=float)

        # beta0
        beta0 = 0.0 if beta0_mode_cfg == "zero" else float(y_curve[0])

        t_cur = float(t_axis[int(cur_idx)])
        y_hist = y_curve[int(cur_idx) - lookback:int(cur_idx)]
        t_hist = t_axis[int(cur_idx) - lookback:int(cur_idx)]

        if len(y_hist) != lookback or np.any(~np.isfinite(y_hist)):
            return np.nan, np.nan, None, "历史窗口不足或包含 NaN/Inf（检查 cur_idx/lookback/数据行）"

        # 历史物理（用于残差/物理特征）
        y_phys_hist = None
        if fusion_mode_cfg in ["残差学习（y = y_phys + Δy_ml）", "物理作为额外特征（Physics as Feature）"]:
            y_phys_hist = compute_y_phys(t_hist, beta0, T, V)

        # 当前是否已失效
        y_now = float(y_curve[int(cur_idx)])
        if fail_rule.startswith("退化量上升") and (y_now >= float(threshold)):
            return 0.0, t_cur, None, "当前时刻已达到阈值（RUL=0）"
        if fail_rule.startswith("健康度下降") and (y_now <= float(threshold)):
            return 0.0, t_cur, None, "当前时刻已达到阈值（RUL=0）"

        pred_times, pred_vals = [], []
        last_t = t_cur

        model.eval()
        with torch.no_grad():
            for _ in range(int(max_roll)):
                t_future = last_t + float(dt) * np.arange(1, horizon + 1, dtype=float)

                y_phys_future = None
                if fusion_mode_cfg == "残差学习（y = y_phys + Δy_ml）":
                    y_phys_future = compute_y_phys(t_future, beta0, T, V)

                X_in = build_x_input(y_hist, t_hist, T, V, y_phys_hist=y_phys_hist)
                out = model(X_in).cpu().numpy().reshape(-1)

                if np.any(~np.isfinite(out)):
                    return np.nan, np.nan, (np.array(pred_times), np.array(pred_vals)), "模型输出 NaN/Inf（训练不稳定或输入不匹配）"

                if fusion_mode_cfg == "残差学习（y = y_phys + Δy_ml）":
                    y_pred = y_phys_future + out
                else:
                    y_pred = out

                pred_times.extend(list(t_future))
                pred_vals.extend(list(y_pred))

                y_pred_arr = np.array(y_pred, dtype=float)

                # 触阈
                if fail_rule.startswith("退化量上升"):
                    hit = np.where(y_pred_arr >= float(threshold))[0]
                else:
                    hit = np.where(y_pred_arr <= float(threshold))[0]

                if len(hit) > 0:
                    k = int(hit[0])
                    t_fail = float(t_future[k])
                    RUL = t_fail - t_cur
                    return float(RUL), float(t_fail), (np.array(pred_times), np.array(pred_vals)), "OK"

                # 滚动更新
                y_ext = np.concatenate([y_hist, y_pred_arr], axis=0)
                t_ext = np.concatenate([t_hist, t_future], axis=0)
                y_hist = y_ext[-lookback:]
                t_hist = t_ext[-lookback:]

                if fusion_mode_cfg in ["残差学习（y = y_phys + Δy_ml）", "物理作为额外特征（Physics as Feature）"]:
                    y_phys_hist = compute_y_phys(t_hist, beta0, T, V)

                last_t = float(t_future[-1])

        # 未触阈：返回诊断信息
        if len(pred_vals) > 0:
            pv_min, pv_max = float(np.min(pred_vals)), float(np.max(pred_vals))
            return np.nan, np.nan, (np.array(pred_times), np.array(pred_vals)), f"外推未触阈（预测范围 y∈[{pv_min:.6g},{pv_max:.6g}]）"
        return np.nan, np.nan, None, "未生成预测（检查 horizon/dt/max_roll）"

    # -----------------------
    # 聚合规则：串联=min，并联=max；混合：Stage 并联=max，Stage 串联=min
    # -----------------------
    def agg_series(ruls):
        ruls = [r for r in ruls if np.isfinite(r)]
        return float(np.min(ruls)) if len(ruls) else np.nan

    def agg_parallel(ruls):
        ruls = [r for r in ruls if np.isfinite(r)]
        return float(np.max(ruls)) if len(ruls) else np.nan

    # -----------------------
    # 点击预测按钮
    # -----------------------
    pred_clicked = st.button("📌 预测电路级 RUL", help="对每个元器件预测 RUL，并按拓扑聚合为电路级 RUL", key="sec63_pred_btn")

    if pred_clicked and comps is not None:
        try:
            plot_cache = {}
            detail_rows = []

            # 逐元器件预测
            for c in comps:
                y_curve = y_df.iloc[int(c["row"]), :].to_numpy(dtype=float)
                RUL, t_fail, traj, status = predict_rul_for_one_curve(
                    y_curve, t_axis, T_use, V_use, int(cur_idx), float(o_thr),
                    max_roll=int(max_roll), dt=float(dt_user)
                )

                t_pred, y_pred = (traj[0], traj[1]) if traj is not None else (None, None)

                plot_cache[c["name"]] = {
                    "row": int(c["row"]),
                    "type": c["type"],
                    "t_pred": t_pred,
                    "y_pred": y_pred,
                    "t_fail": t_fail,
                    "RUL": RUL,
                    "status": status,
                }

                detail_rows.append({
                    "Component": c["name"],
                    "Type": c["type"],
                    "Row": int(c["row"]),
                    "RUL": RUL,
                    "t_fail": t_fail,
                    "status": status
                })

            detail_df = pd.DataFrame(detail_rows)

            # 电路级聚合
            if topo == "串联模型":
                RUL_sys = agg_series(detail_df["RUL"].to_numpy(dtype=float))
                sys_text = f"电路级 RUL（串联，取 min）：{RUL_sys:.6f}"
            elif topo == "并联模型":
                RUL_sys = agg_parallel(detail_df["RUL"].to_numpy(dtype=float))
                sys_text = f"电路级 RUL（并联冗余，取 max）：{RUL_sys:.6f}"
            else:
                # 混合：按 stage 先算 stage_rul，再串联取 min
                stage_rows = []
                stage_ruls = []

                for stg in stages_spec:
                    ruls_this = []
                    for c in stg["comps"]:
                        ruls_this.append(plot_cache[c["name"]]["RUL"])
                    if stg["type"] == "single":
                        R_stage = agg_series(ruls_this)  # 其实就是单个
                    else:
                        R_stage = agg_parallel(ruls_this)

                    stage_ruls.append(R_stage)
                    stage_rows.append({"Stage": stg["stage"], "StageType": stg["type"], "R_stage": R_stage})

                RUL_sys = agg_series(stage_ruls)
                sys_text = f"电路级 RUL（混合：Stage 并联取 max；Stage 串联取 min）：{RUL_sys:.6f}"
                st.write("Stage RUL 明细：")
                st.dataframe(pd.DataFrame(stage_rows))

            # 缓存结果（避免 rerun 丢失）
            st.session_state["sec63_plot_cache"] = plot_cache
            st.session_state["sec63_detail_df"] = detail_df
            st.session_state["sec63_sys_text"] = sys_text
            st.session_state["sec63_meta"] = {
                "topo": topo,
                "cond": int(cond_sel),
                "cur_idx": int(cur_idx),
                "o": float(o_thr),
                "fail_rule": fail_rule,
                "max_roll": int(max_roll),
                "dt": float(dt_user),
            }

        except Exception as e:
            st.error(f"RUL 预测出错：{e}")

    # -----------------------
    # 结果展示（只要缓存存在且参数未变就展示）
    # -----------------------
    meta = st.session_state.get("sec63_meta", None)
    can_show = (
        meta is not None
        and meta.get("topo") == topo
        and meta.get("cond") == int(cond_sel)
        and meta.get("cur_idx") == int(cur_idx)
        and abs(meta.get("o", 0.0) - float(o_thr)) < 1e-12
        and meta.get("fail_rule") == fail_rule
        and meta.get("max_roll") == int(max_roll)
        and abs(meta.get("dt", 0.0) - float(dt_user)) < 1e-12
    )

    if can_show and "sec63_detail_df" in st.session_state:
        st.write("元器件 RUL 预测明细：")
        st.dataframe(st.session_state["sec63_detail_df"])
        st.success(st.session_state.get("sec63_sys_text", ""))

        # NaN 提示
        df_show = st.session_state["sec63_detail_df"]
        if not np.isfinite(df_show["RUL"].to_numpy(dtype=float)).any():
            st.warning("所有元器件均未在外推范围内触阈 → 电路级 RUL 可能为 NaN。优先检查：阈值量级、触发方向、max_roll/dt。")

        # -----------------------
        # 绘图区域
        # -----------------------
        st.markdown("##### 预测曲线可视化（含阈值线）")
        show_plot = st.checkbox("显示预测曲线图", value=True, key="sec63_show_plot")

        plot_cache = st.session_state.get("sec63_plot_cache", {})
        if show_plot and len(plot_cache) > 0:
            comp_to_plot = st.selectbox("选择要绘制的元器件", list(plot_cache.keys()), key="sec63_comp_plot")
            cinfo = plot_cache[comp_to_plot]
            y_curve = _clean_curve(y_df.iloc[int(cinfo["row"]), :].to_numpy(dtype=float))

            fig = plot_rul_prediction(
                t_axis=t_axis,
                y_curve=y_curve,
                cur_idx=int(cur_idx),
                lookback=int(lookback),
                pred_times=cinfo["t_pred"],
                pred_vals=cinfo["y_pred"],
                threshold=float(o_thr),
                t_fail=cinfo["t_fail"],
                RUL=cinfo["RUL"],
                title=f"{comp_to_plot} ({cinfo['type']}) — RUL prediction"
            )
            st.pyplot(fig)

            # 诊断：预测范围 vs 阈值
            if cinfo["y_pred"] is not None and len(cinfo["y_pred"]) > 0:
                st.caption(
                    f"{comp_to_plot} 预测范围：y_pred ∈ [{np.min(cinfo['y_pred']):.6g}, {np.max(cinfo['y_pred']):.6g}]；"
                    f"阈值 o={float(o_thr):.6g}；status={cinfo.get('status','')}"
                )
        elif show_plot:
            st.info("暂无可绘图数据：请先点击上面的“📌 预测电路级 RUL”。")
    else:
        st.info("参数已变更或尚未预测：请点击上面的“📌 预测电路级 RUL”生成结果与图。")
