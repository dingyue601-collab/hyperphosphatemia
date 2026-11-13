"""Observed age and creatinine, frozen TabM inference, no retained records."""
import streamlit as st
from inference import TwoPredictorModels, clinical_input

st.set_page_config(page_title="Type 2 diabetic kidney disease hyperphosphatemia alert", page_icon=":material/nephrology:", layout="centered")
st.session_state.setdefault("unit", "umol/L")
st.session_state.setdefault("previous_unit", "umol/L")
st.session_state.setdefault("age", 65.0)
st.session_state.setdefault("creatinine", 180.0)
st.session_state.setdefault("result", None)
st.session_state.setdefault("auto_calculate", True)


def clear_result():
    st.session_state.result = None
    st.session_state.auto_calculate = False


def convert_unit():
    if st.session_state.unit not in ("umol/L", "mg/dL"):
        st.session_state.unit = st.session_state.previous_unit
        return
    if st.session_state.creatinine is not None and st.session_state.unit != st.session_state.previous_unit:
        factor = 88.4 if st.session_state.unit == "umol/L" else 1 / 88.4
        st.session_state.creatinine *= factor
    st.session_state.previous_unit = st.session_state.unit
    clear_result()


def example():
    st.session_state.age = 65.0
    st.session_state.creatinine = 180.0 if st.session_state.unit == "umol/L" else 180 / 88.4
    clear_result()
    st.session_state.auto_calculate = True


def reset():
    st.session_state.age = None
    st.session_state.creatinine = None
    clear_result()


@st.cache_resource(max_entries=1)
def load_models():
    return TwoPredictorModels(device="cpu")


lang = st.segmented_control("Language", ["中文", "English"], default="English", label_visibility="collapsed", key="language")
zh = lang != "English"
st.title("2型糖尿病肾病高磷血症风险警示" if zh else "Type 2 diabetic kidney disease hyperphosphatemia alert")
st.markdown("适用于成人、非透析的2型糖尿病肾病住院人群。" if zh else
    "For adult, non-dialysis inpatients with type 2 diabetic kidney disease.")
unit = st.segmented_control("肌酐单位" if zh else "Creatinine unit", ["umol/L", "mg/dL"],
    format_func=lambda s: "μmol/L" if s == "umol/L" else s, key="unit", required=True, on_change=convert_unit)
left, right = st.columns(2)
left.number_input("年龄（岁）" if zh else "Age (years)", min_value=18.0, max_value=115.0,
    step=1.0, format="%.1f", key="age", placeholder="请输入" if zh else "Enter age", on_change=clear_result)
right.number_input(("血清肌酐" if zh else "Serum creatinine") + (" (μmol/L)" if unit == "umol/L" else " (mg/dL)"),
    min_value=0.001, step=1.0 if unit == "umol/L" else 0.1, format="%.2f", key="creatinine",
    placeholder="请输入" if zh else "Enter creatinine", on_change=clear_result)
with st.container(horizontal=True):
    calculate = st.button("计算" if zh else "Calculate", type="primary", key="calculate")
    st.button("示例" if zh else "Example", on_click=example, key="example")
    st.button("清空" if zh else "Clear", on_click=reset, key="clear")
if calculate or st.session_state.auto_calculate:
    st.session_state.auto_calculate = False
    try:
        frame = clinical_input(st.session_state.age, st.session_state.creatinine, unit)
        with st.spinner("正在计算…" if zh else "Calculating…"):
            model = load_models()
            st.session_state.result = (frame, model.predict(frame))
    except (ValueError, TypeError):
        st.error("请填写有效年龄与肌酐，两项均不可缺失。" if zh else "Enter valid age and creatinine; neither may be missing.")

if st.session_state.result is not None:
    frame, result = st.session_state.result
    model = load_models()
    probability = float(result["TabM2"][0])
    threshold = model.spec["models"]["TabM2"]["threshold"]
    with st.container(border=True):
        st.metric("TabM · 高磷可能性" if zh else "TabM · Estimated probability", f"{probability:.1%}")
        # Text remains available alongside the probability bar and threshold marker.
        width = min(100.0, max(0.0, 100 * probability))
        marker = min(100.0, max(0.0, 100 * threshold))
        st.html(f"""
          <div style="height:18px;border-radius:12px;background:#e7eceb;position:relative;overflow:hidden;margin:4px 0 5px" aria-hidden="true">
            <div style="width:{width:.6f}%;height:100%;background:linear-gradient(90deg,#8fc4bc,#397d75);border-radius:12px"></div>
            <div style="position:absolute;left:{marker:.6f}%;top:0;height:100%;width:3px;background:#343a40"></div>
          </div>
          <div style="display:flex;justify-content:space-between;color:#73827f;font-size:.8rem"><span>0%</span><span>100%</span></div>
        """)
        st.caption(("筛查阈值 " if zh else "Screening threshold ") + f"{threshold:.2%}")
        st.markdown(("**高磷血症风险警示**" if probability >= threshold else "**未达警示阈值**") if zh else
            ("**Hyperphosphatemia alert**" if probability >= threshold else "**Below alert threshold**"))
    ranges = model.spec["training_observed_range"]
    if any(not ranges[v][0] <= float(frame[v].iloc[0]) <= ranges[v][1] for v in ("age", "creatinine")):
        st.warning("至少一项超出训练观测范围，结果外推不确定。" if zh else "An input is outside the observed training range; extrapolation is uncertain.")

st.subheader("模型信息" if zh else "Model information")
with st.container(border=True):
    st.markdown("**识别目标：** 入院第0或第1天血磷 >1.45 mmol/L。" if zh else
        "**Target:** phosphate >1.45 mmol/L on admission day 0 or 1.")
    st.markdown("**研究人群：** 成人、非透析的2型糖尿病肾病住院患者。" if zh else
        "**Population:** adult, non-dialysis inpatients with type 2 diabetic kidney disease.")
    st.markdown("**输入：** 年龄和血清肌酐。" if zh else
        "**Inputs:** age and serum creatinine.")
    st.markdown("**版本：** [TabM 0.0.3](https://github.com/yandex-research/tabm)，32成员、30模型平均。" if zh else
        "**Version:** [TabM 0.0.3](https://github.com/yandex-research/tabm); an average of 32 members across 30 fitted models.")
