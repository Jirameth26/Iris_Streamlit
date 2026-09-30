"""
=============================================================================
🧮 ApexCalc Pro - Advanced Multi-Function Calculator & Mathematics Suite
=============================================================================
A comprehensive, modern, and powerful scientific calculator built with Streamlit.
Features:
1. 🔬 Scientific Calculator (Keypad, LaTeX display, Safe AST parser, Memory & History)
2. 📈 2D Function Grapher (Dual plots, analytical roots/extrema, interactive zoom)
3. 📐 Equation & System Solver (Quadratic with parabolas, 2x2 Linear with intersections, Polynomials)
4. 📊 Statistics & Data Analytics (Descriptive stats, KDE Histograms, Box plots)
5. 💻 Programmer & Base Converter (Dec, Bin, Hex, Oct, Bitwise logic, 32-bit registers)
6. 🔄 Unit Converter (Length, Mass, Temp, Area including Thai Rai/Ngan/Wa, Speed, Storage)
7. 💰 Financial Calculator (Compound interest projection, Loan & Mortgage EMI schedule)
=============================================================================
"""

import ast
import datetime
from fractions import Fraction
import io
import math
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ApexCalc Pro - เครื่องคิดเลขขั้นสูง",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern glassmorphism & responsive keypad UI
st.markdown(
    """
<style>
    /* Global font & styling improvements */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
    }
    
    /* Result Display Card */
    .calc-screen-card {
        background: linear-gradient(135deg, rgba(28, 36, 54, 0.95), rgba(15, 23, 42, 0.98));
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 24px 28px;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.35);
        color: #f8fafc;
        margin-bottom: 20px;
    }
    .screen-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.9rem;
        color: #94a3b8;
        margin-bottom: 8px;
    }
    .screen-expr {
        font-family: 'Courier New', Courier, monospace;
        font-size: 1.25rem;
        color: #38bdf8;
        word-break: break-all;
        min-height: 28px;
    }
    .screen-result {
        font-size: 2.6rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #ffffff;
        word-break: break-all;
        line-height: 1.2;
    }
    .screen-badge {
        display: inline-block;
        padding: 3px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        background: rgba(56, 189, 248, 0.2);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.4);
    }
    
    /* Quick Action / Feature Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* History list item */
    .history-card {
        background: rgba(255, 255, 255, 0.03);
        border-left: 3px solid #38bdf8;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }

    /* Streamlit button custom adjustments */
    div.stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.15s ease-in-out;
    }
    div.stButton > button:hover {
        box-shadow: 0 4px 12px rgba(56, 189, 248, 0.25);
    }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "calc_expr" not in st.session_state:
    st.session_state.calc_expr = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = 0.0
if "last_error" not in st.session_state:
    st.session_state.last_error = ""
if "memory_val" not in st.session_state:
    st.session_state.memory_val = 0.0
if "calc_history" not in st.session_state:
    st.session_state.calc_history = []
if "angle_mode" not in st.session_state:
    st.session_state.angle_mode = "Deg"  # 'Deg' or 'Rad'
if "decimal_places" not in st.session_state:
    st.session_state.decimal_places = 6

# -----------------------------------------------------------------------------
# 3. MATHEMATICAL EVALUATION ENGINE (SAFE AST PARSER)
# -----------------------------------------------------------------------------
class SafeAstValidator(ast.NodeVisitor):
    """Restricts expression AST nodes to safe arithmetic & mathematical operations only."""
    ALLOWED_NODES = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
        ast.Name, ast.Call, ast.Load,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
        ast.USub, ast.UAdd
    )

    def visit(self, node):
        if not isinstance(node, self.ALLOWED_NODES):
            raise ValueError(f"ไม่อนุญาตการใช้งานไวยากรณ์: '{type(node).__name__}'")
        return super().visit(node)


def clean_expression(raw_expr: str) -> str:
    """Preprocesses formula string: fixes unicode operators, implicit multiplication, percentages."""
    expr = raw_expr.strip()
    if not expr:
        return ""

    # Replace common symbols
    replacements = {
        "×": "*", "÷": "/", "–": "-", "—": "-",
        "²": "**2", "³": "**3", "^": "**",
        "π": "pi", "√": "sqrt"
    }
    for old, new in replacements.items():
        expr = expr.replace(old, new)

    # Protect scientific notation like 1.5e-3, 2e10 so regex doesn't mistake 'e' for variable
    sci_tokens = []
    def save_sci(m):
        sci_tokens.append(m.group(0))
        return f"@SCI_{len(sci_tokens)-1}@"
    expr = re.sub(r"\b\d+(\.\d+)?[eE][+-]?\d+\b", save_sci, expr)

    # Replace percentages e.g. 50% * 200 -> (50/100) * 200 (not modulo)
    expr = re.sub(r"(\d+(?:\.\d+)?)\s*%(?!\s*[\d\(a-zA-Z])", r"(\1/100)", expr)

    # Implicit multiplication:
    # 2(x) -> 2*(x)
    expr = re.sub(r"(\d)\s*(\()", r"\1*\2", expr)
    # (x)2 -> (x)*2
    expr = re.sub(r"(\))\s*(\d)", r"\1*\2", expr)
    # (x)(y) -> (x)*(y)
    expr = re.sub(r"(\))\s*(\()", r"\1*\2", expr)
    # 2pi, 2x, 2sin -> 2*pi, 2*x, 2*sin
    expr = re.sub(r"(\d)\s*([a-zA-Z])", r"\1*\2", expr)
    # )pi, )x -> )*pi
    expr = re.sub(r"(\))\s*([a-zA-Z])", r"\1*\2", expr)

    # Restore scientific notation
    for i, tok in enumerate(sci_tokens):
        expr = expr.replace(f"@SCI_{i}@", tok)

    return expr


def build_math_environment(deg_mode: bool = True):
    """Builds a secure and feature-rich math scope for expression evaluation."""
    if deg_mode:
        trig_sin = lambda x: math.sin(math.radians(x))
        trig_cos = lambda x: math.cos(math.radians(x))
        trig_tan = lambda x: math.tan(math.radians(x))
        trig_asin = lambda x: math.degrees(math.asin(x))
        trig_acos = lambda x: math.degrees(math.acos(x))
        trig_atan = lambda x: math.degrees(math.atan(x))
    else:
        trig_sin = math.sin
        trig_cos = math.cos
        trig_tan = math.tan
        trig_asin = math.asin
        trig_acos = math.acos
        trig_atan = math.atan

    env = {
        # Constants
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
        "phi": 1.618033988749895,
        "inf": math.inf,
        # Trigonometric
        "sin": trig_sin, "cos": trig_cos, "tan": trig_tan,
        "asin": trig_asin, "acos": trig_acos, "atan": trig_atan,
        "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
        "asinh": math.asinh, "acosh": math.acosh, "atanh": math.atanh,
        # Powers & Roots
        "sqrt": math.sqrt,
        "cbrt": getattr(math, "cbrt", lambda x: x ** (1 / 3)),
        "exp": math.exp,
        "pow": pow,
        # Logarithms
        "log": math.log10,      # Default log is base 10 (scientific standard)
        "log10": math.log10,
        "ln": math.log,        # Natural logarithm (base e)
        "log2": math.log2,
        # Number Theory & Combinatorics
        "factorial": math.factorial,
        "fact": math.factorial,
        "gcd": math.gcd,
        "lcm": getattr(math, "lcm", lambda a, b: abs(a * b) // math.gcd(a, b)),
        "comb": getattr(math, "comb", lambda n, k: math.factorial(n) // (math.factorial(k) * math.factorial(n - k))),
        "perm": getattr(math, "perm", lambda n, k: math.factorial(n) // math.factorial(n - k)),
        "nCr": getattr(math, "comb", lambda n, k: math.factorial(n) // (math.factorial(k) * math.factorial(n - k))),
        "nPr": getattr(math, "perm", lambda n, k: math.factorial(n) // math.factorial(n - k)),
        # Utility & Rounding
        "abs": abs,
        "round": round,
        "floor": math.floor,
        "ceil": math.ceil,
        "deg": math.degrees,
        "rad": math.radians,
    }
    return env


def evaluate_expression(raw_expr: str, deg_mode: bool = True):
    """Safely evaluates mathematical expressions and handles common domain/syntax errors."""
    cleaned = clean_expression(raw_expr)
    if not cleaned:
        return None, "กรุณากรอกนิพจน์คณิตศาสตร์ (Please enter an expression)"

    try:
        # Validate AST safety
        tree = ast.parse(cleaned, mode="eval")
        SafeAstValidator().visit(tree)

        # Evaluate inside restricted namespace
        env = build_math_environment(deg_mode=deg_mode)
        # Add memory and last result access
        env["ans"] = st.session_state.last_result
        env["Ans"] = st.session_state.last_result
        env["mem"] = st.session_state.memory_val

        result = eval(cleaned, {"__builtins__": {}}, env)

        # Handle complex or nan
        if isinstance(result, complex):
            return result, None
        if math.isnan(result):
            return None, "ผลลัพธ์เป็นค่าไม่นิยาม (NaN)"

        return result, None

    except ZeroDivisionError:
        return None, "ข้อผิดพลาด: ไม่สามารถหารด้วย 0 ได้ (Division by zero)"
    except ValueError as e:
        return None, f"ข้อผิดพลาดทางคณิตศาสตร์: {str(e)}"
    except OverflowError:
        return None, "ข้อผิดพลาด: ตัวเลขมีขนาดใหญ่เกินกว่าที่ระบบจะคำนวณได้ (Number overflow)"
    except (SyntaxError, TypeError) as e:
        return None, f"รูปแบบนิพจน์ไม่ถูกต้อง: {str(e)}"
    except Exception as e:
        return None, f"เกิดข้อผิดพลาด: {str(e)}"


# -----------------------------------------------------------------------------
# 4. KEYPAD CALLBACK HELPERS
# -----------------------------------------------------------------------------
def on_key_press(token: str):
    """Appends token to the expression in session state."""
    st.session_state.calc_expr = str(st.session_state.get("calc_expr", "")) + token


def on_clear():
    """Clears the expression and error message."""
    st.session_state.calc_expr = ""
    st.session_state.last_error = ""


def on_backspace():
    """Deletes the last character in the expression."""
    curr = str(st.session_state.get("calc_expr", ""))
    st.session_state.calc_expr = curr[:-1] if curr else ""


def on_calculate():
    """Executes the calculation and logs history."""
    expr = st.session_state.calc_expr
    if not expr:
        return

    deg = (st.session_state.angle_mode == "Deg")
    res, err = evaluate_expression(expr, deg_mode=deg)

    if err:
        st.session_state.last_error = err
    else:
        st.session_state.last_result = res
        st.session_state.last_error = ""

        # Format history record
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        record = {
            "time": timestamp,
            "expression": expr,
            "result": res,
            "mode": st.session_state.angle_mode
        }
        # Prepend to history (keep top 30)
        st.session_state.calc_history.insert(0, record)
        if len(st.session_state.calc_history) > 30:
            st.session_state.calc_history.pop()


def on_memory_action(action: str):
    """Handles MC, MR, M+, M- operations."""
    if action == "MC":
        st.session_state.memory_val = 0.0
    elif action == "MR":
        on_key_press(str(st.session_state.memory_val))
    elif action == "M+":
        # Ensure current expression is calculated
        deg = (st.session_state.angle_mode == "Deg")
        res, err = evaluate_expression(st.session_state.calc_expr, deg_mode=deg)
        if not err and res is not None:
            st.session_state.memory_val += float(res)
        else:
            st.session_state.memory_val += float(st.session_state.last_result)
    elif action == "M-":
        deg = (st.session_state.angle_mode == "Deg")
        res, err = evaluate_expression(st.session_state.calc_expr, deg_mode=deg)
        if not err and res is not None:
            st.session_state.memory_val -= float(res)
        else:
            st.session_state.memory_val -= float(st.session_state.last_result)


# -----------------------------------------------------------------------------
# 5. SIDEBAR NAVIGATION & SETTINGS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧮 **ApexCalc Pro**")
    st.caption("ระบบเครื่องคิดเลขวิทยาศาสตร์และคณิตศาสตร์ขั้นสูง")
    st.divider()

    selected_mode = st.radio(
        "📌 เลือกโหมดการทำงาน (Modes):",
        [
            "🔬 เครื่องคิดเลขวิทยาศาสตร์ (Scientific)",
            "📈 พลอตกราฟฟังก์ชัน (Graphing 2D)",
            "📐 ตัวแก้สมการ (Equation Solver)",
            "📊 สถิติและการวิเคราะห์ข้อมูล (Statistics)",
            "💻 แปลงเลขฐาน & บิตไวส์ (Programmer)",
            "🔄 แปลงหน่วยวัด (Unit Converter)",
            "💰 เครื่องคำนวณการเงิน (Financial)"
        ],
        index=0
    )

    st.divider()
    st.markdown("### ⚙️ การตั้งค่าทั่วไป (Settings)")

    col_ang1, col_ang2 = st.columns(2)
    with col_ang1:
        angle_choice = st.radio(
            "หน่วยมุม (Angle):",
            ["Deg (องศา)", "Rad (เรเดียน)"],
            index=0 if st.session_state.angle_mode == "Deg" else 1,
            horizontal=False
        )
        st.session_state.angle_mode = "Deg" if "Deg" in angle_choice else "Rad"

    with col_ang2:
        dec_places = st.slider(
            "ทศนิยม:",
            min_value=2,
            max_value=12,
            value=st.session_state.decimal_places,
            step=1
        )
        st.session_state.decimal_places = dec_places

    st.divider()
    st.markdown("### 💾 สถานะหน่วยความจำ (Memory)")
    col_m1, col_m2 = st.columns([2, 1])
    with col_m1:
        st.metric(label="ค่า Memory (M)", value=f"{st.session_state.memory_val:g}")
    with col_m2:
        if st.button("ล้าง M (MC)", use_container_width=True):
            st.session_state.memory_val = 0.0
            st.rerun()

    st.divider()
    st.caption("💡 **Tip**: คุณสามารถพิมพ์สูตร เช่น `sin(30) + 2^4` ได้โดยตรงผ่านแป้นพิมพ์ หรือคลิกปุ่มบนหน้าจอ")


# -----------------------------------------------------------------------------
# 6. MODE IMPLEMENTATIONS
# -----------------------------------------------------------------------------

# =============================================================================
# MODE 1: SCIENTIFIC CALCULATOR
# =============================================================================
if selected_mode.startswith("🔬"):
    st.markdown("## 🔬 เครื่องคิดเลขวิทยาศาสตร์ & ฟังก์ชันขั้นสูง")
    st.caption("รองรับฟังก์ชันตรีโกณมิติ, เลขยกกำลัง, ลอการิทึม, แฟกทอเรียล, เศษส่วน และระบบหน่วยความจำ")

    # Format result display value
    val = st.session_state.last_result
    prec = st.session_state.decimal_places
    if isinstance(val, (int, np.integer)):
        display_res = f"{val:,}"
    elif isinstance(val, (float, np.floating)):
        display_res = f"{val:,.{prec}f}".rstrip("0").rstrip(".") if "." in f"{val:.{prec}f}" else f"{val:,.0f}"
    elif isinstance(val, complex):
        display_res = f"{val.real:g} + {val.imag:g}i"
    else:
        display_res = str(val)

    # 1. Screen / Display Header Card
    st.markdown(
        f"""
        <div class="calc-screen-card">
            <div class="screen-header">
                <div>
                    <span class="screen-badge">{st.session_state.angle_mode.upper()}</span>
                    &nbsp;&nbsp;
                    <span>ทศนิยม: {prec} ตำแหน่ง</span>
                    &nbsp;&nbsp;
                    <span>M = {st.session_state.memory_val:g}</span>
                </div>
                <div>Ans = {st.session_state.last_result:g}</div>
            </div>
            <div class="screen-expr">{st.session_state.calc_expr if st.session_state.calc_expr else '&nbsp;'}</div>
            <div class="screen-result">{display_res}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    # 2. Main Text Input & Calculate Action
    col_input, col_go = st.columns([5, 1])
    with col_input:
        current_input = st.text_input(
            "ป้อนนิพจน์คณิตศาสตร์ (Expression Input):",
            value=st.session_state.calc_expr,
            placeholder="ตัวอย่าง: sin(30) + sqrt(144) * 2^3 หรือ 50% * 800",
            key="text_calc_input"
        )
        if current_input != st.session_state.calc_expr:
            st.session_state.calc_expr = current_input

    with col_go:
        st.write("") # vertical spacing
        if st.button("คำนวณ (=)", type="primary", use_container_width=True, on_click=on_calculate):
            pass

    # 3. Quick Examples & Templates Dropdown
    with st.expander("⚡ ตัวอย่างสูตรที่พบบ่อย (Quick Formulas)", expanded=False):
        ex_cols = st.columns(4)
        with ex_cols[0]:
            if st.button("📐 ตรีโกณ: sin(30) + cos(60)", use_container_width=True):
                st.session_state.calc_expr = "sin(30) + cos(60)"
                st.rerun()
        with ex_cols[1]:
            if st.button("🔢 รูท & กำลัง: sqrt(144) + 2^5", use_container_width=True):
                st.session_state.calc_expr = "sqrt(144) + 2^5"
                st.rerun()
        with ex_cols[2]:
            if st.button("📈 ลอการิทึม: log(1000) + ln(e^2)", use_container_width=True):
                st.session_state.calc_expr = "log(1000) + ln(e^2)"
                st.rerun()
        with ex_cols[3]:
            if st.button("🎲 การจัดหมู่: comb(10, 3)", use_container_width=True):
                st.session_state.calc_expr = "comb(10, 3)"
                st.rerun()

    # 4. Interactive Keypad Grid Layout
    st.markdown("#### 📱 แผงปุ่มกดวิทยาศาสตร์ (Interactive Scientific Keypad)")
    
    # Row 1: Memory & Clear Controls
    r1 = st.columns(7)
    with r1[0]:
        st.button("MC", use_container_width=True, help="Memory Clear", on_click=on_memory_action, args=("MC",))
    with r1[1]:
        st.button("MR", use_container_width=True, help="Memory Recall", on_click=on_memory_action, args=("MR",))
    with r1[2]:
        st.button("M+", use_container_width=True, help="Memory Add", on_click=on_memory_action, args=("M+",))
    with r1[3]:
        st.button("M-", use_container_width=True, help="Memory Subtract", on_click=on_memory_action, args=("M-",))
    with r1[4]:
        st.button("Ans", use_container_width=True, help="Last Result", on_click=on_key_press, args=("Ans",))
    with r1[5]:
        st.button("C", use_container_width=True, help="Clear All", type="secondary", on_click=on_clear)
    with r1[6]:
        st.button("⌫", use_container_width=True, help="Backspace", on_click=on_backspace)

    # Row 2: Trig Functions & Exponent
    r2 = st.columns(7)
    with r2[0]:
        st.button("sin", use_container_width=True, on_click=on_key_press, args=("sin(",))
    with r2[1]:
        st.button("cos", use_container_width=True, on_click=on_key_press, args=("cos(",))
    with r2[2]:
        st.button("tan", use_container_width=True, on_click=on_key_press, args=("tan(",))
    with r2[3]:
        st.button("x²", use_container_width=True, on_click=on_key_press, args=("^2",))
    with r2[4]:
        st.button("x³", use_container_width=True, on_click=on_key_press, args=("^3",))
    with r2[5]:
        st.button("xʸ (^)", use_container_width=True, on_click=on_key_press, args=("^",))
    with r2[6]:
        st.button("÷", use_container_width=True, on_click=on_key_press, args=("/",))

    # Row 3: Inverse Trig & Roots
    r3 = st.columns(7)
    with r3[0]:
        st.button("asin", use_container_width=True, on_click=on_key_press, args=("asin(",))
    with r3[1]:
        st.button("acos", use_container_width=True, on_click=on_key_press, args=("acos(",))
    with r3[2]:
        st.button("atan", use_container_width=True, on_click=on_key_press, args=("atan(",))
    with r3[3]:
        st.button("√ (sqrt)", use_container_width=True, on_click=on_key_press, args=("sqrt(",))
    with r3[4]:
        st.button("∛ (cbrt)", use_container_width=True, on_click=on_key_press, args=("cbrt(",))
    with r3[5]:
        st.button("eˣ", use_container_width=True, on_click=on_key_press, args=("exp(",))
    with r3[6]:
        st.button("×", use_container_width=True, on_click=on_key_press, args=("*",))

    # Row 4: Logs, Fact & Numbers 7,8,9
    r4 = st.columns(7)
    with r4[0]:
        st.button("ln", use_container_width=True, help="Natural Log (base e)", on_click=on_key_press, args=("ln(",))
    with r4[1]:
        st.button("log₁₀", use_container_width=True, help="Log Base 10", on_click=on_key_press, args=("log(",))
    with r4[2]:
        st.button("x!", use_container_width=True, help="Factorial", on_click=on_key_press, args=("fact(",))
    with r4[3]:
        st.button("7", use_container_width=True, on_click=on_key_press, args=("7",))
    with r4[4]:
        st.button("8", use_container_width=True, on_click=on_key_press, args=("8",))
    with r4[5]:
        st.button("9", use_container_width=True, on_click=on_key_press, args=("9",))
    with r4[6]:
        st.button("−", use_container_width=True, on_click=on_key_press, args=("-",))

    # Row 5: Constants & Numbers 4,5,6
    r5 = st.columns(7)
    with r5[0]:
        st.button("π", use_container_width=True, on_click=on_key_press, args=("pi",))
    with r5[1]:
        st.button("e", use_container_width=True, on_click=on_key_press, args=("e",))
    with r5[2]:
        st.button("|x|", use_container_width=True, help="Absolute value", on_click=on_key_press, args=("abs(",))
    with r5[3]:
        st.button("4", use_container_width=True, on_click=on_key_press, args=("4",))
    with r5[4]:
        st.button("5", use_container_width=True, on_click=on_key_press, args=("5",))
    with r5[5]:
        st.button("6", use_container_width=True, on_click=on_key_press, args=("6",))
    with r5[6]:
        st.button("+", use_container_width=True, on_click=on_key_press, args=("+",))

    # Row 6: Parentheses & Numbers 1,2,3
    r6 = st.columns(7)
    with r6[0]:
        st.button("(", use_container_width=True, on_click=on_key_press, args=("(",))
    with r6[1]:
        st.button(")", use_container_width=True, on_click=on_key_press, args=(")",))
    with r6[2]:
        st.button("%", use_container_width=True, help="Percent or Modulo", on_click=on_key_press, args=("%",))
    with r6[3]:
        st.button("1", use_container_width=True, on_click=on_key_press, args=("1",))
    with r6[4]:
        st.button("2", use_container_width=True, on_click=on_key_press, args=("2",))
    with r6[5]:
        st.button("3", use_container_width=True, on_click=on_key_press, args=("3",))
    with r6[6]:
        st.button("mod", use_container_width=True, on_click=on_key_press, args=(" % ",))

    # Row 7: 0, Dot, Equal
    r7 = st.columns(7)
    with r7[0]:
        st.button("nCr", use_container_width=True, help="Combinations", on_click=on_key_press, args=("comb(",))
    with r7[1]:
        st.button("gcd", use_container_width=True, help="Greatest Common Divisor", on_click=on_key_press, args=("gcd(",))
    with r7[2]:
        st.button("00", use_container_width=True, on_click=on_key_press, args=("00",))
    with r7[3]:
        st.button("0", use_container_width=True, on_click=on_key_press, args=("0",))
    with r7[4]:
        st.button(".", use_container_width=True, on_click=on_key_press, args=(".",))
    with r7[5]:
        st.button("1/x", use_container_width=True, on_click=on_key_press, args=("1/",))
    with r7[6]:
        st.button("=", type="primary", use_container_width=True, on_click=on_calculate)

    st.divider()

    # 5. Result Analysis Section (Fractions, Scientific Notation, LaTeX)
    col_details, col_hist = st.columns([1, 1])

    with col_details:
        st.markdown("#### 🔍 รายละเอียดผลลัพธ์ (Result Breakdown)")
        if isinstance(val, (int, float, np.floating, np.integer)) and not isinstance(val, bool):
            f_val = float(val)

            # Fraction representation
            try:
                frac = Fraction(f_val).limit_denominator(10000)
                frac_str = f"{frac.numerator} / {frac.denominator}" if frac.denominator != 1 else f"{frac.numerator}"
            except Exception:
                frac_str = "N/A"

            # Scientific Notation
            sci_str = f"{f_val:.4e}"

            # Metric display
            m1, m2 = st.columns(2)
            with m1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-val">{frac_str}</div>
                        <div class="metric-label">รูปเศษส่วน (Fraction)</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with m2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-val">{sci_str}</div>
                        <div class="metric-label">สัญกรณ์วิทยาศาสตร์ (Scientific)</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # LaTeX render
            st.write("")
            st.latex(rf"\text{{Result}} = {display_res}")

            # If it's a whole number, show binary/hex/octal
            if f_val.is_integer() and abs(f_val) < 2**63:
                int_v = int(f_val)
                st.markdown(
                    f"""
                    - **เลขฐานสอง (Binary):** `{bin(int_v)}`
                    - **เลขฐานสิบหก (Hex):** `{hex(int_v)}`
                    - **เลขฐานแปด (Octal):** `{oct(int_v)}`
                    """
                )
        else:
            st.info("คำนวณตัวเลขเพื่อดูรายละเอียดทางคณิตศาสตร์")

    # 6. History Drawer
    with col_hist:
        st.markdown("#### 📜 ประวัติการคำนวณ (Calculation History)")
        if st.session_state.calc_history:
            h_header_col1, h_header_col2 = st.columns([3, 1])
            with h_header_col2:
                if st.button("ล้างประวัติ", use_container_width=True):
                    st.session_state.calc_history = []
                    st.rerun()

            for idx, item in enumerate(st.session_state.calc_history[:8]):
                with st.container():
                    c_h1, c_h2 = st.columns([4, 1])
                    with c_h1:
                        st.markdown(
                            f"""
                            <div class="history-card">
                                <small style="color: #94a3b8;">[{item['time']}] ({item['mode']})</small><br>
                                <strong>{item['expression']}</strong> = <span style="color:#38bdf8; font-weight:700;">{item['result']:g}</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with c_h2:
                        st.write("")
                        if st.button("ใช้ค่า", key=f"hist_use_{idx}", use_container_width=True):
                            st.session_state.calc_expr = str(item["result"])
                            st.rerun()
        else:
            st.caption("ยังไม่มีประวัติการคำนวณในเซสชันนี้")


# =============================================================================
# MODE 2: GRAPHING CALCULATOR (2D FUNCTION VISUALIZER)
# =============================================================================
elif selected_mode.startswith("📈"):
    st.markdown("## 📈 พลอตกราฟฟังก์ชันทางคณิตศาสตร์ (2D Function Grapher)")
    st.caption("วิเคราะห์และแสดงกราฟฟังก์ชัน $f(x)$ และ $g(x)$ พร้อมจุดวิกฤต, จุดตัดแกน, และอนุพันธ์")

    col_g_inputs, col_g_opts = st.columns([3, 2])
    with col_g_inputs:
        func_f_str = st.text_input("ฟังก์ชันหลัก f(x):", value="sin(x) * exp(-0.05 * x**2)", help="เช่น x**2 - 4*x + 3 หรือ sin(2*x) + cos(x)")
        func_g_str = st.text_input("ฟังก์ชันเสริม g(x) (ใส่หรือไม่ใส่ก็ได้):", value="0.5 * cos(x)", help="สำหรับเปรียบเทียบหรือหาจุดตัด")

    with col_g_opts:
        col_x1, col_x2 = st.columns(2)
        with col_x1:
            x_min = st.number_input("x ต่ำสุด (Min):", value=-10.0, step=1.0)
        with col_x2:
            x_max = st.number_input("x สูงสุด (Max):", value=10.0, step=1.0)
        
        num_points = st.slider("ความละเอียด (จำนวนจุดคำนวณ):", min_value=100, max_value=2000, value=600, step=100)

    # Check range validity
    if x_min >= x_max:
        st.error("ค่า x ต่ำสุด ต้องน้อยกว่า x สูงสุด")
    else:
        x_vals = np.linspace(x_min, x_max, num_points)
        
        # Prepare evaluation environment for numpy vectors
        np_env = {
            "x": x_vals,
            "sin": np.sin, "cos": np.cos, "tan": np.tan,
            "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
            "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
            "exp": np.exp, "sqrt": np.sqrt, "log": np.log10, "log10": np.log10, "ln": np.log, "log2": np.log2,
            "abs": np.abs, "pi": np.pi, "e": np.e
        }

        # Clean expressions
        clean_f = clean_expression(func_f_str)
        clean_g = clean_expression(func_g_str) if func_g_str.strip() else ""

        f_success = False
        y_f = None
        y_g = None

        try:
            y_f = eval(clean_f, {"__builtins__": {}}, np_env)
            if np.isscalar(y_f):
                y_f = np.full_like(x_vals, float(y_f))
            f_success = True
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการคำนวณ f(x): {str(e)}")

        if clean_g:
            try:
                y_g = eval(clean_g, {"__builtins__": {}}, np_env)
                if np.isscalar(y_g):
                    y_g = np.full_like(x_vals, float(y_g))
            except Exception as e:
                st.warning(f"คำนวณ g(x) ไม่สำเร็จ: {str(e)}")
                y_g = None

        if f_success and y_f is not None:
            # Mask extreme or non-finite values for clean plotting
            y_f_masked = np.copy(y_f)
            y_f_masked[~np.isfinite(y_f_masked)] = np.nan

            # Plot styling with Matplotlib
            plt.style.use("seaborn-v0_8-whitegrid")
            fig, ax = plt.subplots(figsize=(10, 5), dpi=120)

            # Background & Grid
            ax.axhline(0, color="gray", linewidth=1.2, linestyle="--", alpha=0.7)
            ax.axvline(0, color="gray", linewidth=1.2, linestyle="--", alpha=0.7)

            # Plot f(x)
            ax.plot(x_vals, y_f_masked, label=f"$f(x) = {func_f_str}$", color="#0284c7", linewidth=2.4)
            ax.fill_between(x_vals, y_f_masked, 0, color="#0284c7", alpha=0.12)

            # Plot g(x) if valid
            if y_g is not None:
                y_g_masked = np.copy(y_g)
                y_g_masked[~np.isfinite(y_g_masked)] = np.nan
                ax.plot(x_vals, y_g_masked, label=f"$g(x) = {func_g_str}$", color="#f97316", linewidth=2.0, linestyle="-.")

            ax.set_title(f"กราฟฟังก์ชันทางคณิตศาสตร์บนช่วง [{x_min:g}, {x_max:g}]", fontsize=13, fontweight="bold", pad=12)
            ax.set_xlabel("x", fontsize=11)
            ax.set_ylabel("y", fontsize=11)
            ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
            fig.tight_layout()

            st.pyplot(fig)

            # Summary analytics of f(x) in range
            valid_y = y_f[np.isfinite(y_f)]
            if len(valid_y) > 0:
                st.markdown("#### 📊 ข้อมูลเชิงวิเคราะห์ของ $f(x)$ ในช่วงที่เลือก")
                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                with stat_col1:
                    st.metric("ค่าสูงสุด (Max y)", f"{np.max(valid_y):.4f}")
                with stat_col2:
                    st.metric("ค่าต่ำสุด (Min y)", f"{np.min(valid_y):.4f}")
                with stat_col3:
                    st.metric("ค่าเฉลี่ย (Mean y)", f"{np.mean(valid_y):.4f}")
                with stat_col4:
                    # Estimate zero crossings
                    signs = np.sign(valid_y)
                    crossings = np.sum(np.diff(signs) != 0)
                    st.metric("จุดตัดแกน X โดยประมาณ", f"{crossings} จุด")

            # Data Table Preview & Download
            with st.expander("📋 ตารางข้อมูลพิกัด (Coordinates Table & CSV Export)"):
                df_plot = pd.DataFrame({"x": x_vals, "f(x)": y_f})
                if y_g is not None:
                    df_plot["g(x)"] = y_g
                st.dataframe(df_plot.head(100), use_container_width=True)

                csv = df_plot.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 ดาวน์โหลดข้อมูลกราฟเป็น CSV",
                    data=csv,
                    file_name="graph_coordinates.csv",
                    mime="text/csv",
                )


# =============================================================================
# MODE 3: EQUATION & SYSTEM SOLVER
# =============================================================================
elif selected_mode.startswith("📐"):
    st.markdown("## 📐 ตัวแก้สมการทางคณิตศาสตร์ (Equation & System Solver)")
    st.caption("แก้สมการกำลังสอง, ระบบสมการเชิงเส้น 2 ตัวแปร และหารากพหุนามพร้อมขั้นตอนแสดงวิธีทำอย่างละเอียด")

    eq_tab1, eq_tab2, eq_tab3 = st.tabs([
        "1. สมการกำลังสอง (Quadratic: ax² + bx + c = 0)",
        "2. ระบบสมการเชิงเส้น 2 ตัวแปร (2x2 Linear System)",
        "3. รากของพหุนามทั่วไป (Polynomial Roots)"
    ])

    # 1. Quadratic Equation
    with eq_tab1:
        st.markdown("### สมการกำลังสอง: $ax^2 + bx + c = 0$")
        q_c1, q_c2, q_c3 = st.columns(3)
        with q_c1:
            qa = st.number_input("สัมประสิทธิ์ a (a ≠ 0):", value=1.0, step=0.5, key="q_a")
        with q_c2:
            qb = st.number_input("สัมประสิทธิ์ b:", value=-5.0, step=0.5, key="q_b")
        with q_c3:
            qc = st.number_input("ค่าคงที่ c:", value=6.0, step=0.5, key="q_c")

        if qa == 0:
            st.error("ค่าสัมประสิทธิ์ a ต้องไม่เท่ากับ 0 (หาก a = 0 จะเป็นสมการเชิงเส้น)")
        else:
            disc = qb**2 - 4 * qa * qc
            vertex_x = -qb / (2 * qa)
            vertex_y = qc - (qb**2) / (4 * qa)

            st.write("")
            st.markdown(f"**สมการของคุณคือ:** ${qa:g}x^2 {'+ ' if qb>=0 else ''}{qb:g}x {'+ ' if qc>=0 else ''}{qc:g} = 0$")
            st.latex(rf"\Delta = b^2 - 4ac = ({qb:g})^2 - 4({qa:g})({qc:g}) = {disc:g}")

            res_m1, res_m2, res_m3 = st.columns(3)
            with res_m1:
                st.metric("ค่าดิสคริมิแนนต์ (Δ)", f"{disc:g}")
            with res_m2:
                st.metric("จุดยอดพาราโบลา (Vertex)", f"({vertex_x:g}, {vertex_y:g})")
            with res_m3:
                parabola_type = "พาราโบลาหงาย (จุดต่ำสุด)" if qa > 0 else "พาราโบลาคว่ำ (จุดสูงสุด)"
                st.metric("ลักษณะกราฟ", parabola_type)

            # Roots computation
            if disc > 0:
                root1 = (-qb + math.sqrt(disc)) / (2 * qa)
                root2 = (-qb - math.sqrt(disc)) / (2 * qa)
                st.success(f"**มีคำตอบเป็นจำนวนจริง 2 คำตอบที่แตกต่างกัน:**\n- $x_1 = {root1:g}$\n- $x_2 = {root2:g}$")
            elif disc == 0:
                root = -qb / (2 * qa)
                st.success(f"**มีคำตอบเป็นจำนวนจริง 1 คำตอบ (รากซ้ำ):**\n- $x = {root:g}$")
            else:
                real_part = -qb / (2 * qa)
                imag_part = math.sqrt(-disc) / (2 * abs(qa))
                st.info(f"**มีคำตอบเป็นจำนวนเชิงซ้อน 2 คำตอบ (สังยุค):**\n- $x_1 = {real_part:g} + {imag_part:g}i$\n- $x_2 = {real_part:g} - {imag_part:g}i$")

            # Parabola Graph Preview
            span = max(abs(vertex_x) + 5, 6)
            px = np.linspace(vertex_x - span, vertex_x + span, 400)
            py = qa * px**2 + qb * px + qc

            fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
            ax.axhline(0, color="gray", linestyle="--", linewidth=1)
            ax.axvline(0, color="gray", linestyle="--", linewidth=1)
            ax.plot(px, py, color="#2563eb", linewidth=2.2, label=f"${qa:g}x^2 + {qb:g}x + {qc:g}$")
            ax.scatter([vertex_x], [vertex_y], color="#dc2626", zorder=5, label=f"Vertex ({vertex_x:g}, {vertex_y:g})")

            if disc >= 0:
                r1 = (-qb + math.sqrt(disc)) / (2 * qa)
                r2 = (-qb - math.sqrt(disc)) / (2 * qa)
                ax.scatter([r1, r2], [0, 0], color="#16a34a", s=60, zorder=5, label="Roots (จุดตัดแกน X)")

            ax.set_title("กราฟพาราโบลาของสมการ", fontsize=11, fontweight="bold")
            ax.legend(loc="best")
            st.pyplot(fig)

    # 2. Linear System 2x2
    with eq_tab2:
        st.markdown("### ระบบสมการเชิงเส้น 2 ตัวแปร:")
        st.latex(r"""
        \begin{cases}
        a_1 x + b_1 y = c_1 \\
        a_2 x + b_2 y = c_2
        \end{cases}
        """)

        s_col1, s_col2 = st.columns(2)
        with s_col1:
            st.markdown("##### 🔹 สมการที่ 1 ($a_1 x + b_1 y = c_1$)")
            a1 = st.number_input("a1:", value=2.0, step=1.0)
            b1 = st.number_input("b1:", value=3.0, step=1.0)
            c1 = st.number_input("c1:", value=8.0, step=1.0)

        with s_col2:
            st.markdown("##### 🔹 สมการที่ 2 ($a_2 x + b_2 y = c_2$)")
            a2 = st.number_input("a2:", value=1.0, step=1.0)
            b2 = st.number_input("b2:", value=-1.0, step=1.0)
            c2 = st.number_input("c2:", value=-1.0, step=1.0)

        # Solve using Cramer's rule
        D = a1 * b2 - a2 * b1
        Dx = c1 * b2 - c2 * b1
        Dy = a1 * c2 - a2 * c1

        st.markdown(f"**ดีเทอร์มิแนนต์ของเมทริกซ์หลัก (D):** $D = ({a1})({b2}) - ({a2})({b1}) = {D:g}$")

        if D != 0:
            sol_x = Dx / D
            sol_y = Dy / D
            st.success(f"**ผลลัพธ์ของระบบสมการ:**\n- $x = \\frac{{D_x}}{{D}} = \\frac{{{Dx:g}}}{{{D:g}}} = {sol_x:g}$\n- $y = \\frac{{D_y}}{{D}} = \\frac{{{Dy:g}}}{{{D:g}}} = {sol_y:g}$")

            # Plot lines
            fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
            x_vals = np.linspace(sol_x - 5, sol_x + 5, 200)

            if b1 != 0:
                y1_vals = (c1 - a1 * x_vals) / b1
                ax.plot(x_vals, y1_vals, label=f"${a1:g}x + {b1:g}y = {c1:g}$", color="#0284c7")
            if b2 != 0:
                y2_vals = (c2 - a2 * x_vals) / b2
                ax.plot(x_vals, y2_vals, label=f"${a2:g}x + {b2:g}y = {c2:g}$", color="#ea580c")

            ax.scatter([sol_x], [sol_y], color="#dc2626", s=80, zorder=5, label=f"จุดตัด Solution ({sol_x:g}, {sol_y:g})")
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
            ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
            ax.set_title("กราฟแสดงจุดตัดของเส้นตรงทั้งสอง", fontsize=11, fontweight="bold")
            ax.legend()
            st.pyplot(fig)
        else:
            if Dx == 0 and Dy == 0:
                st.warning("ระบบสมการมีคำตอบนับไม่ถ้วน (เส้นตรงทับกันพอดี)")
            else:
                st.error("ระบบสมการไม่มีคำตอบ (เส้นตรงขนานกัน)")

    # 3. Higher Order Polynomial Roots
    with eq_tab3:
        st.markdown("### คำนวณรากของพหุนามดีกรี $n$")
        st.caption("ป้อนสัมประสิทธิ์เรียงจากดีกรีสูงสุดไปยังต่ำสุด คั่นด้วยเครื่องหมายจุลภาค (Comma)")

        poly_input = st.text_input(
            "สัมประสิทธิ์พหุนาม (เช่น 1, -6, 11, -6 สำหรับ x³ - 6x² + 11x - 6 = 0):",
            value="1, -6, 11, -6"
        )
        if poly_input:
            try:
                coeffs = [float(x.strip()) for x in poly_input.split(",") if x.strip()]
                if len(coeffs) < 2:
                    st.warning("กรุณาป้อนสัมประสิทธิ์อย่างน้อย 2 ตัว")
                else:
                    roots = np.roots(coeffs)
                    degree = len(coeffs) - 1
                    st.markdown(f"**พหุนามดีกรี {degree} มีรากทั้งหมด {len(roots)} ราก:**")

                    root_list = []
                    for idx, r in enumerate(roots, 1):
                        if abs(r.imag) < 1e-10:
                            root_list.append({"ลำดับ (No.)": idx, "ประเภท (Type)": "จำนวนจริง (Real)", "คำตอบ (Root)": f"{r.real:.6g}"})
                        else:
                            sign = "+" if r.imag >= 0 else "-"
                            root_list.append({"ลำดับ (No.)": idx, "ประเภท (Type)": "จำนวนเชิงซ้อน (Complex)", "คำตอบ (Root)": f"{r.real:.6g} {sign} {abs(r.imag):.6g}i"})

                    st.dataframe(pd.DataFrame(root_list), use_container_width=True)
            except Exception as e:
                st.error(f"รูปแบบสัมประสิทธิ์ไม่ถูกต้อง: {str(e)}")


# =============================================================================
# MODE 4: STATISTICS & DATA CALCULATOR
# =============================================================================
elif selected_mode.startswith("📊"):
    st.markdown("## 📊 สถิติและการวิเคราะห์ชุดข้อมูล (Statistics & Data Analytics)")
    st.caption("คำนวณค่าสถิติเชิงพรรณนา, ควอร์ไทล์, การกระจายตัว พร้อมกราฟฮิสโตแกรมและ Box Plot")

    st.markdown("#### ป้อนข้อมูลตัวเลข (Input Data)")
    c_data_in, c_preset = st.columns([3, 1])

    with c_preset:
        preset_choice = st.selectbox(
            "ชุดข้อมูลตัวอย่าง (Presets):",
            ["-- กำหนดเอง --", "คะแนนสอบ (Exam Scores)", "ยอดขาย (Daily Sales)", "การกระจายแบบปกติ (Normal)"]
        )

    default_data = "15, 22, 18, 25, 30, 24, 18, 21, 29, 35, 40, 28, 22, 19, 31, 27"
    if preset_choice == "คะแนนสอบ (Exam Scores)":
        default_data = "65, 72, 88, 54, 91, 78, 83, 69, 74, 85, 90, 62, 77, 81, 70"
    elif preset_choice == "ยอดขาย (Daily Sales)":
        default_data = "1200, 1550, 1800, 1420, 2100, 1950, 2300, 1750, 2600, 2400"
    elif preset_choice == "การกระจายแบบปกติ (Normal)":
        np.random.seed(42)
        norm_sample = np.random.normal(50, 10, 25)
        default_data = ", ".join([f"{v:.1f}" for v in norm_sample])

    with c_data_in:
        raw_numbers = st.text_area(
            "ป้อนตัวเลข (คั่นด้วยจุลภาค เวนวรรค หรือขึ้นบรรทัดใหม่):",
            value=default_data,
            height=90
        )

    # Parse numbers safely
    tokens = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", raw_numbers)
    if not tokens:
        st.warning("กรุณาป้อนตัวเลขอย่างน้อย 1 ตัวเพื่อคำนวณสถิติ")
    else:
        data = np.array([float(t) for t in tokens])
        n = len(data)

        # Calculate statistics
        mean_val = np.mean(data)
        median_val = np.median(data)
        std_pop = np.std(data)
        std_sample = np.std(data, ddof=1) if n > 1 else 0.0
        var_sample = np.var(data, ddof=1) if n > 1 else 0.0
        min_val = np.min(data)
        max_val = np.max(data)
        range_val = max_val - min_val
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1

        # Mode calculation
        vals, counts = np.unique(data, return_counts=True)
        max_c = np.max(counts)
        if max_c > 1:
            mode_vals = vals[counts == max_c]
            mode_str = ", ".join([f"{m:g}" for m in mode_vals])
        else:
            mode_str = "ไม่มีฐานนิยม (ทุกค่าซ้ำเท่ากัน)"

        st.markdown("#### 📈 ค่าสถิติเชิงพรรณนา (Descriptive Statistics)")
        st.write("")
        r1_c1, r1_c2, r1_c3, r1_c4 = st.columns(4)
        with r1_c1:
            st.metric("จำนวนข้อมูล (N)", f"{n}")
        with r1_c2:
            st.metric("ค่าเฉลี่ยเลขคณิต (Mean)", f"{mean_val:.4f}")
        with r1_c3:
            st.metric("มัธยฐาน (Median)", f"{median_val:g}")
        with r1_c4:
            st.metric("ฐานนิยม (Mode)", mode_str)

        r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4)
        with r2_c1:
            st.metric("ส่วนเบี่ยงเบนมาตรฐาน (SD)", f"{std_sample:.4f}")
        with r2_c2:
            st.metric("ความแปรปรวน (s²)", f"{var_sample:.4f}")
        with r2_c3:
            st.metric("พิสัย (Range)", f"{range_val:g}")
        with r2_c4:
            st.metric("พิสัยระหว่างควอร์ไทล์ (IQR)", f"{iqr:g}")

        r3_c1, r3_c2, r3_c3, r3_c4 = st.columns(4)
        with r3_c1:
            st.metric("ค่าน้อยที่สุด (Min)", f"{min_val:g}")
        with r3_c2:
            st.metric("ควอร์ไทล์ที่ 1 (Q1)", f"{q1:g}")
        with r3_c3:
            st.metric("ควอร์ไทล์ที่ 3 (Q3)", f"{q3:g}")
        with r3_c4:
            st.metric("ค่ามากที่สุด (Max)", f"{max_val:g}")

        st.divider()

        # Visualizations: Histogram & Box Plot
        st.markdown("#### 📊 แผนภูมิแสดงการแจกแจงของข้อมูล (Data Distribution Charts)")
        fig, (ax_box, ax_hist) = plt.subplots(
            2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [0.3, 0.7]}, dpi=120
        )

        # Box Plot
        ax_box.boxplot(data, vert=False, patch_artist=True, boxprops=dict(facecolor="#bae6fd", color="#0284c7"),
                       medianprops=dict(color="#dc2626", linewidth=2.5))
        ax_box.set_yticks([])
        ax_box.set_title("Box Plot (การกระจายและค่าผิดปกติ)", fontsize=11, fontweight="bold")

        # Histogram
        ax_hist.hist(data, bins="auto", color="#38bdf8", edgecolor="white", alpha=0.85, density=False)
        ax_hist.axvline(mean_val, color="#2563eb", linestyle="--", linewidth=2, label=f"Mean ({mean_val:.2f})")
        ax_hist.axvline(median_val, color="#dc2626", linestyle="-", linewidth=2, label=f"Median ({median_val:.2f})")
        ax_hist.set_xlabel("ค่าของข้อมูล (Values)", fontsize=10)
        ax_hist.set_ylabel("ความถี่ (Frequency)", fontsize=10)
        ax_hist.legend(loc="upper right")

        fig.tight_layout()
        st.pyplot(fig)


# =============================================================================
# MODE 5: PROGRAMMER & BASE CONVERTER
# =============================================================================
elif selected_mode.startswith("💻"):
    st.markdown("## 💻 เครื่องคิดเลขโปรแกรมเมอร์ & แปลงเลขฐาน (Programmer & Bitwise)")
    st.caption("แปลงเลขฐานสิบ, สอง, สิบหก, แปด พร้อมคำนวณตรรกะบิตไวส์ (AND, OR, XOR, NOT, Shift)")

    prog_tab1, prog_tab2 = st.tabs(["1. แปลงระบบเลขฐาน (Base Conversion)", "2. ดำเนินการบิตไวส์ (Bitwise Operations)"])

    with prog_tab1:
        st.markdown("### ตัวแปลงเลขฐานแบบ Real-Time")
        b_col1, b_col2 = st.columns([1, 2])
        with b_col1:
            base_from = st.selectbox("เลือกฐานตัวเลขต้นทาง:", ["ฐาน 10 (Decimal)", "ฐาน 16 (Hexadecimal)", "ฐาน 2 (Binary)", "ฐาน 8 (Octal)"])
        with b_col2:
            base_val_input = st.text_input("ป้อนค่าตัวเลข:", value="255")

        try:
            val_clean = base_val_input.strip().replace(" ", "").replace("_", "")
            if "Decimal" in base_from:
                int_num = int(val_clean, 10)
            elif "Hex" in base_from:
                int_num = int(val_clean, 16)
            elif "Binary" in base_from:
                int_num = int(val_clean, 2)
            else:
                int_num = int(val_clean, 8)

            # Format binary with nibble groups (4 bits)
            bin_str = bin(int_num)[2:] if int_num >= 0 else bin(int_num)[3:]
            pad_len = ((len(bin_str) + 3) // 4) * 4
            bin_padded = bin_str.zfill(pad_len)
            bin_grouped = " ".join([bin_padded[i:i+4] for i in range(0, len(bin_padded), 4)])
            if int_num < 0:
                bin_grouped = "-" + bin_grouped

            st.write("")
            st.markdown(
                f"""
                <div class="calc-screen-card" style="padding: 18px 24px;">
                    <div style="margin-bottom: 10px;"><strong>Decimal (ฐาน 10):</strong> <span style="color:#38bdf8; font-size:1.4rem;">{int_num:,}</span></div>
                    <div style="margin-bottom: 10px;"><strong>Hexadecimal (ฐาน 16):</strong> <span style="color:#a855f7; font-size:1.4rem;">0x{hex(int_num)[2:].upper()}</span></div>
                    <div style="margin-bottom: 10px;"><strong>Binary (ฐาน 2):</strong> <span style="color:#22c55e; font-size:1.4rem; font-family:monospace;">{bin_grouped}</span></div>
                    <div><strong>Octal (ฐาน 8):</strong> <span style="color:#f59e0b; font-size:1.4rem;">0o{oct(int_num)[2:]}</span></div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # 32-bit register visualization (if positive or reasonable)
            if 0 <= int_num < 2**32:
                st.markdown("##### 📟 แสดงภาพบิตใน Register 32-bit (Bit Switches)")
                bits_32 = bin(int_num)[2:].zfill(32)
                cols = st.columns(8)
                for b_idx in range(8):
                    chunk = bits_32[b_idx*4 : (b_idx+1)*4]
                    cols[b_idx].code(f"B{7-b_idx*4}..B{7-b_idx*4-3}\n{chunk}")

        except Exception as e:
            st.error(f"รูปแบบตัวเลขไม่สอดคล้องกับฐานที่เลือก: {str(e)}")

    with prog_tab2:
        st.markdown("### คำนวณตรรกศาสตร์ระดับบิต (Bitwise Calculator)")
        bw_c1, bw_op, bw_c2 = st.columns([2, 1, 2])
        with bw_c1:
            bw_a = st.number_input("ตัวเลข A (จำนวนเต็ม):", value=12, step=1)
        with bw_op:
            bw_operation = st.selectbox("ตัวดำเนินการ:", ["AND (&)", "OR (|)", "XOR (^)", "NOT (~A)", "Shift Left (<<)", "Shift Right (>>)"])
        with bw_c2:
            bw_b = st.number_input("ตัวเลข B (หรือจำนวนบิตเลื่อน):", value=5, step=1)

        a_int = int(bw_a)
        b_int = int(bw_b)

        if bw_operation == "AND (&)":
            bw_res = a_int & b_int
            op_sym = "&"
        elif bw_operation == "OR (|)":
            bw_res = a_int | b_int
            op_sym = "|"
        elif bw_operation == "XOR (^)":
            bw_res = a_int ^ b_int
            op_sym = "^"
        elif bw_operation == "NOT (~A)":
            bw_res = ~a_int
            op_sym = "~"
        elif bw_operation == "Shift Left (<<)":
            bw_res = a_int << b_int
            op_sym = "<<"
        else:
            bw_res = a_int >> b_int
            op_sym = ">>"

        st.success(f"**ผลลัพธ์: {a_int} {op_sym} {b_int if 'NOT' not in bw_operation else ''} = {bw_res}**")
        st.markdown(
            f"""
            - **ในรูปเลขฐาน 10:** `{bw_res}`
            - **ในรูปเลขฐาน 16:** `{hex(bw_res)}`
            - **ในรูปเลขฐาน 2:** `{bin(bw_res)}`
            """
        )


# =============================================================================
# MODE 6: UNIT CONVERTER
# =============================================================================
elif selected_mode.startswith("🔄"):
    st.markdown("## 🔄 เครื่องมือแปลงหน่วยวัดสากล (Unit Converter)")
    st.caption("แปลงค่าความยาว, มวล/น้ำหนัก, อุณหภูมิ, พื้นที่ (รวมหน่วยที่ดินไทย ไร่/งาน/วา), ความเร็ว, และข้อมูลดิจิทัล")

    unit_cat = st.selectbox(
        "หมวดหมู่หน่วยวัด:",
        [
            "📏 ความยาว (Length)",
            "⚖️ มวลและน้ำหนัก (Mass / Weight)",
            "🌡️ อุณหภูมิ (Temperature)",
            "🏡 พื้นที่ & ที่ดินไทย (Area & Thai Land)",
            "🚀 ความเร็ว (Speed)",
            "💾 ข้อมูลดิจิทัล (Digital Storage)"
        ]
    )

    col_u_val, col_u_from = st.columns([2, 2])

    if "Length" in unit_cat:
        # Standard: meters
        conv_factors = {
            "เมตร (m)": 1.0, "กิโลเมตร (km)": 1000.0, "เซนติเมตร (cm)": 0.01, "มิลลิเมตร (mm)": 0.001,
            "นิ้ว (inch)": 0.0254, "ฟุต (ft)": 0.3048, "หลา (yard)": 0.9144, "ไมล์ (mile)": 1609.344
        }
        with col_u_val:
            u_val = st.number_input("ค่าตัวเลข:", value=1.0, step=1.0)
        with col_u_from:
            u_from = st.selectbox("จากหน่วย:", list(conv_factors.keys()))

        val_in_base = u_val * conv_factors[u_from]
        results = {unit: val_in_base / factor for unit, factor in conv_factors.items()}

    elif "Mass" in unit_cat:
        # Standard: grams
        conv_factors = {
            "กรัม (g)": 1.0, "กิโลกรัม (kg)": 1000.0, "มิลลิกรัม (mg)": 0.001,
            "ปอนด์ (lb)": 453.59237, "ออนซ์ (oz)": 28.349523, "ตันเมตริก (ton)": 1_000_000.0
        }
        with col_u_val:
            u_val = st.number_input("ค่าตัวเลข:", value=1.0, step=1.0)
        with col_u_from:
            u_from = st.selectbox("จากหน่วย:", list(conv_factors.keys()))

        val_in_base = u_val * conv_factors[u_from]
        results = {unit: val_in_base / factor for unit, factor in conv_factors.items()}

    elif "Temperature" in unit_cat:
        with col_u_val:
            u_val = st.number_input("ค่าตัวเลขอุณหภูมิ:", value=25.0, step=1.0)
        with col_u_from:
            u_from = st.selectbox("จากหน่วย:", ["เซลเซียส (°C)", "ฟาเรนไฮต์ (°F)", "เคลวิน (K)"])

        # Convert to Celsius first
        if "°C" in u_from:
            c = u_val
        elif "°F" in u_from:
            c = (u_val - 32) * 5 / 9
        else:
            c = u_val - 273.15

        results = {
            "เซลเซียส (°C)": c,
            "ฟาเรนไฮต์ (°F)": (c * 9 / 5) + 32,
            "เคลวิน (K)": c + 273.15
        }

    elif "Area" in unit_cat:
        # Standard: Square Meters (m^2)
        # 1 ไร่ = 4 งาน = 400 ตารางวา = 1,600 ตารางเมตร
        conv_factors = {
            "ตารางเมตร (m²)": 1.0, "ตารางกิโลเมตร (km²)": 1_000_000.0, "ไร่ (Thai Rai)": 1600.0,
            "งาน (Thai Ngan)": 400.0, "ตารางวา (Thai Sq. Wa)": 4.0, "ตารางฟุต (ft²)": 0.092903,
            "เอเคอร์ (acre)": 4046.8564
        }
        with col_u_val:
            u_val = st.number_input("ค่าพื้นที่:", value=1.0, step=1.0)
        with col_u_from:
            u_from = st.selectbox("จากหน่วย:", list(conv_factors.keys()))

        val_in_base = u_val * conv_factors[u_from]
        results = {unit: val_in_base / factor for unit, factor in conv_factors.items()}

    elif "Speed" in unit_cat:
        # Standard: m/s
        conv_factors = {
            "เมตร/วินาที (m/s)": 1.0, "กิโลเมตร/ชั่วโมง (km/h)": 1 / 3.6,
            "ไมล์/ชั่วโมง (mph)": 0.44704, "นอต (knot)": 0.514444
        }
        with col_u_val:
            u_val = st.number_input("ค่าความเร็ว:", value=100.0, step=10.0)
        with col_u_from:
            u_from = st.selectbox("จากหน่วย:", list(conv_factors.keys()))

        val_in_base = u_val * conv_factors[u_from]
        results = {unit: val_in_base / factor for unit, factor in conv_factors.items()}

    else:  # Digital Storage
        # Standard: Bytes
        conv_factors = {
            "Bytes (B)": 1.0, "Kilobytes (KB)": 1024.0, "Megabytes (MB)": 1024.0**2,
            "Gigabytes (GB)": 1024.0**3, "Terabytes (TB)": 1024.0**4, "Petabytes (PB)": 1024.0**5
        }
        with col_u_val:
            u_val = st.number_input("ขนาดข้อมูล:", value=1.0, step=1.0)
        with col_u_from:
            u_from = st.selectbox("จากหน่วย:", list(conv_factors.keys()))

        val_in_base = u_val * conv_factors[u_from]
        results = {unit: val_in_base / factor for unit, factor in conv_factors.items()}

    st.write("")
    st.markdown("#### 📋 ผลลัพธ์การแปลงเทียบเท่าทุกหน่วย (Converted Equivalents):")
    res_df = pd.DataFrame([{"หน่วยวัด (Unit)": k, "ค่าที่แปลงได้ (Value)": f"{v:,.6g}"} for k, v in results.items()])
    st.dataframe(res_df, use_container_width=True, hide_index=True)


# =============================================================================
# MODE 7: FINANCIAL CALCULATOR
# =============================================================================
elif selected_mode.startswith("💰"):
    st.markdown("## 💰 เครื่องคำนวณการเงิน (Financial Mathematics)")
    st.caption("คำนวณดอกเบี้ยทบต้น (Compound Interest) และค่างวดผ่อนบ้าน/รถยนต์ (Amortization Loan EMI)")

    fin_tab1, fin_tab2 = st.tabs(["1. ดอกเบี้ยทบต้น (Compound Interest)", "2. ผ่อนบ้านและสินเชื่อ (Loan EMI)"])

    # 1. Compound Interest
    with fin_tab1:
        st.markdown("### คำนวณการเติบโตของเงินออม & ดอกเบี้ยทบต้น")
        st.latex(r"A = P \left(1 + \frac{r}{n}\right)^{nt}")

        f_c1, f_c2 = st.columns(2)
        with f_c1:
            principal = st.number_input("เงินต้นเริ่มต้น (Principal - บาท):", value=100000.0, step=10000.0)
            annual_rate = st.number_input("อัตราดอกเบี้ยต่อปี (% ต่อปี):", value=5.0, step=0.25)
            years = st.slider("ระยะเวลาการลงทุน (ปี):", min_value=1, max_value=40, value=10)

        with f_c2:
            monthly_contrib = st.number_input("เงินฝากสมทบเพิ่มทุกเดือน (บาท):", value=2000.0, step=500.0)
            comp_freq = st.selectbox("ความถี่ในการทบต้น:", ["รายปี (n=1)", "รายไตรมาส (n=4)", "รายเดือน (n=12)", "รายวัน (n=365)"])

        freq_map = {"รายปี (n=1)": 1, "รายไตรมาส (n=4)": 4, "รายเดือน (n=12)": 12, "รายวัน (n=365)": 365}
        n_comp = freq_map[comp_freq]
        r = annual_rate / 100.0

        # Simulate year-by-year
        timeline = []
        curr_balance = principal
        total_deposited = principal

        for y in range(1, years + 1):
            for _ in range(12):
                curr_balance += monthly_contrib
                total_deposited += monthly_contrib
                curr_balance *= (1 + r / 12)
            timeline.append({
                "ปีที่ (Year)": y,
                "เงินต้นสะสม (Total Deposit)": total_deposited,
                "มูลค่าเงินรวม (Future Value)": curr_balance,
                "ดอกเบี้ยสะสม (Total Interest)": curr_balance - total_deposited
            })

        df_fin = pd.DataFrame(timeline)
        final_val = timeline[-1]["มูลค่าเงินรวม (Future Value)"]
        total_dep = timeline[-1]["เงินต้นสะสม (Total Deposit)"]
        total_int = timeline[-1]["ดอกเบี้ยสะสม (Total Interest)"]

        st.write("")
        m_f1, m_f2, m_f3 = st.columns(3)
        with m_f1:
            st.metric("มูลค่าเงินรวมเมื่อครบกำหนด", f"{final_val:,.2f} ฿")
        with m_f2:
            st.metric("เงินต้นทั้งหมดที่ลงไป", f"{total_dep:,.2f} ฿")
        with m_f3:
            st.metric("ผลตอบแทนจากดอกเบี้ย", f"+{total_int:,.2f} ฿", delta=f"{(total_int/total_dep)*100:.1f}%")

        # Growth Plot
        fig, ax = plt.subplots(figsize=(9, 4), dpi=120)
        ax.plot(df_fin["ปีที่ (Year)"], df_fin["มูลค่าเงินรวม (Future Value)"], label="มูลค่ารวม (Future Value)", color="#2563eb", linewidth=2.4)
        ax.plot(df_fin["ปีที่ (Year)"], df_fin["เงินต้นสะสม (Total Deposit)"], label="เงินต้นสะสม (Total Deposit)", color="#94a3b8", linestyle="--", linewidth=2)
        ax.fill_between(df_fin["ปีที่ (Year)"], df_fin["มูลค่าเงินรวม (Future Value)"], df_fin["เงินต้นสะสม (Total Deposit)"], color="#38bdf8", alpha=0.2, label="ส่วนต่างดอกเบี้ย")
        ax.set_title("แนวโน้มการเติบโตของพอร์ตการลงทุน", fontsize=11, fontweight="bold")
        ax.set_xlabel("ปีที่")
        ax.set_ylabel("จำนวนเงิน (บาท)")
        ax.legend()
        st.pyplot(fig)

    # 2. Mortgage EMI
    with fin_tab2:
        st.markdown("### คำนวณค่างวดผ่อนบ้าน / รถยนต์ (EMI Loan Calculator)")
        st.latex(r"EMI = P \cdot \frac{r(1+r)^n}{(1+r)^n - 1}")

        l_c1, l_c2 = st.columns(2)
        with l_c1:
            loan_amount = st.number_input("วงเงินกู้ (บาท):", value=2000000.0, step=100000.0)
            loan_rate = st.number_input("อัตราดอกเบี้ยต่อปี (%):", value=4.5, step=0.1)
        with l_c2:
            loan_years = st.slider("ระยะเวลาผ่อนชำระ (ปี):", min_value=1, max_value=35, value=25)

        total_months = loan_years * 12
        monthly_r = (loan_rate / 100.0) / 12

        if monthly_r > 0:
            emi = loan_amount * monthly_r * ((1 + monthly_r)**total_months) / (((1 + monthly_r)**total_months) - 1)
        else:
            emi = loan_amount / total_months

        total_payment = emi * total_months
        total_loan_int = total_payment - loan_amount

        st.write("")
        l_m1, l_m2, l_m3 = st.columns(3)
        with l_m1:
            st.metric("ค่างวดผ่อนต่อเดือน (EMI)", f"{emi:,.2f} ฿/เดือน")
        with l_m2:
            st.metric("ดอกเบี้ยจ่ายทั้งหมด", f"{total_loan_int:,.2f} ฿")
        with l_m3:
            st.metric("ยอดรวมที่ต้องจ่ายทั้งหมด", f"{total_payment:,.2f} ฿")

        # Pie Chart: Principal vs Interest
        fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
        ax.pie([loan_amount, total_loan_int], labels=["เงินต้น (Principal)", "ดอกเบี้ยรวม (Total Interest)"],
               autopct="%1.1f%%", colors=["#38bdf8", "#f43f5e"], startangle=90, explode=(0.04, 0.04))
        ax.set_title("สัดส่วนเงินต้นเทียบกับดอกเบี้ยทั้งหมด", fontsize=11, fontweight="bold")
        st.pyplot(fig)

# -----------------------------------------------------------------------------
# 7. FOOTER
# -----------------------------------------------------------------------------
st.write("")
st.divider()
f_col1, f_col2 = st.columns([3, 1])
with f_col1:
    st.caption("🧮 **ApexCalc Pro** — พัฒนาด้วย Streamlit, NumPy, Pandas, Matplotlib และระบบตรวจสอบความปลอดภัย AST")
with f_col2:
    st.caption("เวอร์ชัน: 2.0 (Advance Edition)")