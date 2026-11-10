import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 获取当前脚本所在目录并添加到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 导入模型加载函数和预测函数
from model_api import predict, loaded_features, loaded_outputs


# 兼容不同版本的Streamlit
def rerun_app():
    """兼容不同版本的Streamlit rerun方法"""
    try:
        st.rerun()
    except AttributeError:
        try:
            st.experimental_rerun()
        except AttributeError:
            st.write("")
            st.stop()


def dataframe_with_width(df, width=True, **kwargs):
    """兼容不同版本Streamlit的dataframe显示"""
    try:
        return st.dataframe(df, use_container_width=width, **kwargs)
    except TypeError:
        try:
            return st.dataframe(df, **kwargs)
        except Exception:
            return st.write(df)


# ==================== 页面配置 ====================
st.set_page_config(
    page_title="污泥-低阶煤-秸秆协同利用物质流-能量流分析模型 (Attention-LSTM)",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== 初始化session state ====================
if 'page' not in st.session_state:
    st.session_state.page = 'input'
if 'result' not in st.session_state:
    st.session_state.result = None
if 'input_values' not in st.session_state:
    st.session_state.input_values = None
if 'actual_values' not in st.session_state:
    st.session_state.actual_values = {name: None for name in loaded_outputs}
if 'actual_updated' not in st.session_state:
    st.session_state.actual_updated = False

# ==================== 自定义CSS样式 ====================
st.markdown("""
<style>
    html, body, .stApp { font-size: 16px !important; }
    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #e6edf5 100%);
    }
    .main-title {
        color: #1a3a5c; font-size: 2.8rem !important; font-weight: bold;
        text-align: center; margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-title {
        text-align: center; color: #5a7a9a;
        font-size: 1.2rem !important; margin-bottom: 1.5rem; line-height: 1.8;
    }
    .module-title {
        color: #1a3a5c; font-size: 1.5rem !important; font-weight: bold;
        margin-bottom: 1rem; padding: 0.5rem 1rem;
        background: linear-gradient(135deg, rgba(30,144,255,0.1) 0%, rgba(30,144,255,0.05) 100%);
        border-radius: 10px; border-left: 4px solid #1E90FF;
    }
    .stNumberInput { margin-bottom: 0.8rem !important; }
    .stNumberInput label { font-size: 0.95rem !important; font-weight: 600 !important; color: #1a3a5c !important; }
    .stNumberInput input { font-size: 1.1rem !important; padding: 0.5rem 0.8rem !important;
                           border-radius: 8px !important; border: 2px solid #d0d8e0 !important; }
    .stButton>button {
        background: linear-gradient(135deg, #1E90FF 0%, #0066CC 100%);
        color: white !important; font-weight: bold !important; font-size: 1.3rem !important;
        border-radius: 12px !important; border: none !important; width: 100%;
        padding: 0.8rem 2rem !important; transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(30,144,255,0.3); letter-spacing: 1px;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(30,144,255,0.4);
        background: linear-gradient(135deg, #0066CC 0%, #004D99 100%);
    }
    .back-button>button {
        background: linear-gradient(135deg, #6c7a8a 0%, #4a5a6a 100%);
        box-shadow: 0 4px 15px rgba(108,122,138,0.3);
    }
    .back-button>button:hover {
        background: linear-gradient(135deg, #4a5a6a 0%, #3a4a5a 100%);
        box-shadow: 0 8px 25px rgba(108,122,138,0.4);
    }
    .metric-card {
        background: white; padding: 20px 25px; border-radius: 15px; margin-bottom: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08); border-left: 6px solid #1E90FF;
        transition: all 0.3s ease;
    }
    .metric-card:hover { transform: translateX(8px) scale(1.01); box-shadow: 0 8px 30px rgba(0,0,0,0.12); }
    .metric-label { color: #6c7a8a; font-size: 1.0rem !important; font-weight: 600; }
    .metric-value { color: #1a3a5c; font-size: 2.2rem !important; font-weight: bold; margin-top: 5px; }
    .accuracy-card {
        background: white; padding: 15px 20px; border-radius: 12px; margin-bottom: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06); text-align: center; transition: all 0.3s ease;
    }
    .accuracy-card:hover { transform: translateY(-3px); box-shadow: 0 8px 25px rgba(0,0,0,0.1); }
    .accuracy-label { color: #6c7a8a; font-size: 0.95rem !important; font-weight: 500; }
    .accuracy-value { font-size: 1.8rem !important; font-weight: bold; margin-top: 3px; }
    .accuracy-high { color: #32CD32; }
    .accuracy-medium { color: #FFA500; }
    .accuracy-low { color: #FF6B6B; }
    .accuracy-na { color: #8a9aa8; }
    .avg-accuracy-card {
        background: linear-gradient(135deg, #1E90FF 0%, #0066CC 100%);
        padding: 18px 22px; border-radius: 15px; margin-bottom: 10px;
        box-shadow: 0 4px 20px rgba(30,144,255,0.3); text-align: center;
        transition: all 0.3s ease;
    }
    .avg-accuracy-card:hover { transform: translateY(-3px); box-shadow: 0 8px 30px rgba(30,144,255,0.4); }
    .avg-accuracy-label { color: rgba(255,255,255,0.9); font-size: 1.0rem !important; font-weight: 500; }
    .avg-accuracy-value { color: white; font-size: 2.2rem !important; font-weight: bold; margin-top: 3px; }
    .avg-accuracy-sub { color: rgba(255,255,255,0.7); font-size: 0.85rem !important; margin-top: 2px; }
    .actual-input label { font-size: 0.95rem !important; font-weight: 600 !important; color: #1a5c3a !important; }
    hr { margin: 1.5rem 0; border: 0; border-top: 2px solid rgba(30,144,255,0.15); }
    .stAlert { font-size: 1.1rem !important; padding: 0.8rem 1.2rem !important; }
    .streamlit-expanderHeader { font-size: 1.1rem !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)

# ==================== 页面标题 ====================
st.markdown(
    '<h1 class="main-title">🌲 污泥-低阶煤-秸秆协同利用物质流-能量流分析模型</h1>',
    unsafe_allow_html=True
)

# ==================== 输入页面 ====================
if st.session_state.page == 'input':
    st.markdown(
        '<p class="sub-title">本模型基于<strong>Attention-LSTM</strong>算法开发，手动填写<strong>13项</strong>输入参数，点击预测按钮，自动显示<strong>3项</strong>输出参数的预测结果</p>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    # ==================== 模块一：原料比例 ====================
    st.markdown('<div class="module-title">📊 模块一：原料比例</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sludge_ratio = st.number_input(
            '污泥比例（%）', min_value=0.0, max_value=100.0, value=100.0,
            step=1.0, format="%.1f", key="sludge_ratio"
        )
    with col2:
        coal_ratio = st.number_input(
            '低阶煤比例（%）', min_value=0.0, max_value=100.0, value=0.0,
            step=1.0, format="%.1f", key="coal_ratio"
        )
    with col3:
        straw_ratio = st.number_input(
            '秸秆水解残渣比例（%）', min_value=0.0, max_value=100.0, value=0.0,
            step=1.0, format="%.1f", key="straw_ratio"
        )
    with col4:
        binder_ratio = st.number_input(
            '粘结剂比例（%）', min_value=0.0, max_value=100.0, value=0.0,
            step=1.0, format="%.1f", key="binder_ratio"
        )

    st.markdown("---")

    # ==================== 模块二：原料特性 ====================
    st.markdown('<div class="module-title">🔬 模块二：原料特性</div>', unsafe_allow_html=True)
    col5, col6, col7, col8, col9, col10 = st.columns(6)
    with col5:
        c_content = st.number_input(
            'C 含量（%）', min_value=0.0, max_value=100.0, value=37.54,
            step=0.01, format="%.2f", key="c_content"
        )
    with col6:
        n_content = st.number_input(
            'N 含量（%）', min_value=0.0, max_value=100.0, value=6.88,
            step=0.01, format="%.2f", key="n_content"
        )
    with col7:
        s_content = st.number_input(
            'S 含量（%）', min_value=0.0, max_value=100.0, value=1.785,
            step=0.001, format="%.2f", key="s_content"
        )
    with col8:
        moisture = st.number_input(
            '水分含量（%）', min_value=0.0, max_value=100.0, value=1.04,
            step=0.01, format="%.2f", key="moisture"
        )
    with col9:
        volatile = st.number_input(
            '挥发分含量（%）', min_value=0.0, max_value=100.0, value=27.56,
            step=0.01, format="%.2f", key="volatile"
        )
    with col10:
        calorific = st.number_input(
            '热值（MJ/kg）', min_value=0.0, max_value=100.0, value=2.5,
            step=0.01, format="%.2f", key="calorific"
        )

    st.markdown("---")

    # ==================== 模块三：热解工艺参数 ====================
    st.markdown('<div class="module-title">⚙️ 模块三：热解工艺参数</div>', unsafe_allow_html=True)
    col11, col12, col13 = st.columns(3)
    with col11:
        residence_time = st.number_input(
            '停留时间（min）', min_value=0.0, max_value=120.0, value=0.0,
            step=1.0, format="%.0f", key="residence_time"
        )
    with col12:
        heating_rate = st.number_input(
            '升温速率（℃/min）', min_value=0.0, max_value=30.0, value=5.0,
            step=1.0, format="%.0f", key="heating_rate"
        )
    with col13:
        pyrolysis_temp = st.number_input(
            '热解温度（℃）', min_value=200.0, max_value=900.0, value=200.0,
            step=1.0, format="%.0f", key="pyrolysis_temp"
        )

    st.markdown("---")

    # ==================== 预测按钮 ====================
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if st.button("🚀 开始预测", key="predict_btn"):
            # 用字典收集所有 UI 输入
            input_dict = {
                '污泥比例（%）': sludge_ratio,
                '低阶煤比例（%）': coal_ratio,
                '秸秆水解残渣比例（%）': straw_ratio,
                '粘结剂比例（%）': binder_ratio,
                'C 含量（%）': c_content,
                'N 含量（%）': n_content,
                'S 含量（%）': s_content,
                '水分含量（%）': moisture,
                '挥发分含量（%）': volatile,
                '热值（MJ/kg）': calorific,
                '停留时间（min）': residence_time,
                '升温速率（℃/min）': heating_rate,
                '热解温度（℃）': pyrolysis_temp,
            }

            # 按模型训练时的特征顺序构造输入列表
            try:
                input_list = [input_dict[f] for f in loaded_features]
            except KeyError as e:
                st.error(f"❌ UI 缺少模型所需特征: {e}")
                st.stop()

            st.session_state.input_values = dict(zip(loaded_features, input_list))

            with st.spinner("🔄 Attention-LSTM 模型预测中，请稍候..."):
                result = predict(input_list)

                if result['success']:
                    st.session_state.result = result
                    st.session_state.page = 'output'
                    rerun_app()
                else:
                    st.error(f"❌ 预测失败: {result.get('error', '未知错误')}")

    # ==================== 示例数据 ====================
    with st.expander("📖 查看示例输入数据", expanded=False):
        example_data = {
            '参数类别': ['原料比例', '原料比例', '原料比例', '原料比例',
                         '原料特性', '原料特性', '原料特性', '原料特性',
                         '原料特性', '原料特性', '热解工艺', '热解工艺', '热解工艺'],
            '参数名称': ['污泥比例（%）', '低阶煤比例（%）', '秸秆水解残渣比例（%）', '粘结剂比例（%）',
                         'C 含量（%）', 'N 含量（%）', 'S 含量（%）', '水分含量（%）',
                         '挥发分含量（%）', '热值（MJ/kg）', '停留时间（min）',
                         '升温速率（℃/min）', '热解温度（℃）'],
            '示例值': [100, 0, 0, 0, 37.54, 6.88, 1.78, 1.04, 27.56, 2.5, 0, 5, 200],
            '范围': ['0-100', '0-100', '0-100', '0-100', '0-100', '0-100', '0-100',
                     '0-100', '0-100', '0-100', '0-120', '0-30', '200-900']
        }
        df_example = pd.DataFrame(example_data)
        dataframe_with_width(df_example, hide_index=True)

    st.caption("💡 提示：修改输入参数后点击「开始预测」按钮查看结果")

# ==================== 输出页面 ====================
else:
    col_back1, col_back2, col_back3 = st.columns([1, 6, 1])
    with col_back1:
        if st.button("⬅️ 返回", key="back_btn"):
            st.session_state.page = 'input'
            rerun_app()

    st.markdown(
        '<p class="sub-title">手动填写<strong>3项</strong>输出参数的实际值，点击计算准确率按钮，自动显示<strong>3项</strong>输出参数的准确率及平均准确率</p>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    if st.session_state.result and st.session_state.result['success']:
        predictions = st.session_state.result["predictions"]

        # 依据 loaded_outputs 的顺序取值
        out_names = list(loaded_outputs)
        solid_yield = predictions[out_names[0]]
        gas_yield = predictions[out_names[1]]
        total_energy = predictions[out_names[2]]

        # ==================== 模块一：预测结果 ====================
        st.markdown('<div class="module-title">📈 模块一：预测结果</div>', unsafe_allow_html=True)

        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #1E90FF;">
                <div class="metric-label">🔬 {out_names[0]}</div>
                <div class="metric-value" style="color: #1E90FF;">{solid_yield:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with res_col2:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #32CD32;">
                <div class="metric-label">💨 {out_names[1]}</div>
                <div class="metric-value" style="color: #32CD32;">{gas_yield:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with res_col3:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #FF6B6B;">
                <div class="metric-label">⚡ {out_names[2]}</div>
                <div class="metric-value" style="color: #FF6B6B;">{total_energy:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ==================== 模块二：实际值输入 ====================
        st.markdown('<div class="module-title">📊 模块二：实际值（用于计算准确率）</div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:0.95rem; color:#6c7a8a;">💡 请填写输出参数对应的实际值以计算预测准确率，如无实际值可留空（输入0表示无实际值）。</p>',
            unsafe_allow_html=True
        )

        actual_col1, actual_col2, actual_col3 = st.columns(3)
        with actual_col1:
            actual_solid = st.number_input(
                f"实际{out_names[0]}",
                min_value=0.0, max_value=100.0, value=0.0,
                step=0.01, format="%.2f", key="actual_solid"
            )
        with actual_col2:
            actual_gas = st.number_input(
                f"实际{out_names[1]}",
                min_value=0.0, max_value=100.0, value=0.0,
                step=0.01, format="%.2f", key="actual_gas"
            )
        with actual_col3:
            actual_energy = st.number_input(
                f"实际{out_names[2]}",
                min_value=0.0, max_value=100000.0, value=0.0,
                step=0.01, format="%.2f", key="actual_energy"
            )

        if st.button("🔄 计算准确率", key="update_actual"):
            st.session_state.actual_values = {
                out_names[0]: actual_solid if actual_solid > 0 else None,
                out_names[1]: actual_gas if actual_gas > 0 else None,
                out_names[2]: actual_energy if actual_energy > 0 else None,
            }
            st.session_state.actual_updated = True
            st.success("✅ 实际值已更新！")
            rerun_app()

        st.markdown("---")

        # ==================== 模块三：准确率计算 ====================
        st.markdown('<div class="module-title">🎯 模块三：准确率计算</div>', unsafe_allow_html=True)

        def calculate_accuracy(predicted, actual):
            if actual is None or actual == 0:
                return None
            try:
                pred = float(predicted)
                act = float(actual)
                accuracy = (1 - abs(act - pred) / act) * 100
                return max(0, min(100, accuracy))
            except (ValueError, TypeError):
                return None

        def get_accuracy_style(accuracy):
            if accuracy is None:
                return "accuracy-na"
            elif accuracy >= 90:
                return "accuracy-high"
            elif accuracy >= 70:
                return "accuracy-medium"
            else:
                return "accuracy-low"

        def get_accuracy_text(accuracy):
            if accuracy is None:
                return "未提供实际值"
            elif accuracy >= 90:
                return "优秀 ✅"
            elif accuracy >= 70:
                return "良好 ⚠️"
            else:
                return "需改进 ❌"

        actual_values = st.session_state.actual_values
        accuracies = {
            out_names[0]: calculate_accuracy(solid_yield, actual_values.get(out_names[0])),
            out_names[1]: calculate_accuracy(gas_yield, actual_values.get(out_names[1])),
            out_names[2]: calculate_accuracy(total_energy, actual_values.get(out_names[2])),
        }

        valid_accuracies = [acc for acc in accuracies.values() if acc is not None]
        avg_accuracy = sum(valid_accuracies) / len(valid_accuracies) if valid_accuracies else None

        acc_col1, acc_col2, acc_col3, acc_col4 = st.columns(4)

        with acc_col1:
            acc = accuracies.get(out_names[0])
            style_class = get_accuracy_style(acc)
            st.markdown(f"""
            <div class="accuracy-card">
                <div class="accuracy-label">🔬 {out_names[0]}</div>
                <div class="accuracy-value {style_class}">{f'{acc:.2f}%' if acc is not None else 'N/A'}</div>
                <div style="font-size:0.8rem; color:#8a9aa8;">{get_accuracy_text(acc)}</div>
            </div>
            """, unsafe_allow_html=True)

        with acc_col2:
            acc = accuracies.get(out_names[1])
            style_class = get_accuracy_style(acc)
            st.markdown(f"""
            <div class="accuracy-card">
                <div class="accuracy-label">💨 {out_names[1]}</div>
                <div class="accuracy-value {style_class}">{f'{acc:.2f}%' if acc is not None else 'N/A'}</div>
                <div style="font-size:0.8rem; color:#8a9aa8;">{get_accuracy_text(acc)}</div>
            </div>
            """, unsafe_allow_html=True)

        with acc_col3:
            acc = accuracies.get(out_names[2])
            style_class = get_accuracy_style(acc)
            st.markdown(f"""
            <div class="accuracy-card">
                <div class="accuracy-label">⚡ {out_names[2]}</div>
                <div class="accuracy-value {style_class}">{f'{acc:.2f}%' if acc is not None else 'N/A'}</div>
                <div style="font-size:0.8rem; color:#8a9aa8;">{get_accuracy_text(acc)}</div>
            </div>
            """, unsafe_allow_html=True)

        with acc_col4:
            if avg_accuracy is not None:
                st.markdown(f"""
                <div class="avg-accuracy-card">
                    <div class="avg-accuracy-label">📊 平均准确率</div>
                    <div class="avg-accuracy-value">{avg_accuracy:.2f}%</div>
                    <div class="avg-accuracy-sub">基于 {len(valid_accuracies)} 个指标</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="avg-accuracy-card" style="background: linear-gradient(135deg, #8a9aa8 0%, #6c7a8a 100%);">
                    <div class="avg-accuracy-label">📊 平均准确率</div>
                    <div class="avg-accuracy-value">N/A</div>
                    <div class="avg-accuracy-sub">请提供至少一个实际值</div>
                </div>
                """, unsafe_allow_html=True)

        # ==================== 准确率图表 ====================
        st.markdown("---")
        st.markdown("### 📊 准确率可视化")

        chart_data = []
        labels = []
        colors = []
        for name, acc in accuracies.items():
            labels.append(name)
            if acc is not None:
                chart_data.append(acc)
                if acc >= 90:
                    colors.append('#32CD32')
                elif acc >= 70:
                    colors.append('#FFA500')
                else:
                    colors.append('#FF6B6B')
            else:
                chart_data.append(0)
                colors.append('#d0d8e0')

        if any(acc is not None for acc in accuracies.values()):
            fig = go.Figure(data=[
                go.Bar(
                    x=labels,
                    y=chart_data,
                    marker_color=colors,
                    text=[f'{val:.2f}%' if val > 0 else 'N/A' for val in chart_data],
                    textposition='outside',
                    textfont=dict(size=14, color='#1a3a5c'),
                    hovertemplate='<b>%{x}</b><br>准确率: %{text}<extra></extra>'
                )
            ])

            fig.update_layout(
                title=dict(text='各项输出参数的预测准确率', font=dict(size=18, color='#1a3a5c')),
                xaxis=dict(title='输出参数', title_font=dict(size=14, color='#6c7a8a'),
                           tickfont=dict(size=12, color='#1a3a5c')),
                yaxis=dict(title='准确率 (%)', title_font=dict(size=14, color='#6c7a8a'),
                           tickfont=dict(size=12, color='#1a3a5c'),
                           range=[0, 105], gridcolor='rgba(200,200,200,0.2)'),
                height=400,
                plot_bgcolor='rgba(255,255,255,0.8)',
                paper_bgcolor='rgba(255,255,255,0.5)',
                showlegend=False,
                margin=dict(l=50, r=50, t=50, b=50)
            )
            fig.add_hline(y=90, line_dash="dash", line_color="#32CD32", opacity=0.5,
                          annotation_text="优秀线 (90%)", annotation_position="right")
            fig.add_hline(y=70, line_dash="dash", line_color="#FFA500", opacity=0.5,
                          annotation_text="良好线 (70%)", annotation_position="right")

            st.plotly_chart(fig, use_container_width=True)

        # ==================== 输入参数详情 ====================
        with st.expander("📥 查看输入参数详情", expanded=False):
            input_dict = st.session_state.input_values
            if input_dict:
                cols = st.columns(3)
                for idx, (k, v) in enumerate(input_dict.items()):
                    with cols[idx % 3]:
                        st.metric(label=k, value=f"{v:.4f}")

        # ==================== 准确率计算说明 ====================
        st.markdown("---")
        st.markdown("""
        **💡 准确率计算说明：**
        - 准确率计算公式：预测准确度 = (1 - |实际值 - 预测值| / 实际值) × 100%
        - 准确率越高表示预测越准确
        - 🟢 绿色（≥90%）：优秀
        - 🟡 橙色（70%-90%）：良好
        - 🔴 红色（<70%）：需要改进
        - 📌 输入0表示无实际值，不会计算准确率
        """)

        st.caption(f"⏱️ 预测时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

    else:
        st.error("❌ 未找到预测结果，请返回输入页面重新预测")
        if st.button("返回输入页面", key="go_back"):
            st.session_state.page = 'input'
            rerun_app()