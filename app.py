import os
import re
import sys
import io
import warnings
import logging
import asyncio
import streamlit as st
from streamlit.components.v1 import html

# --- Hide Streamlit Branding ---
html("""
<script>
try {
    const sel = window.top.document.querySelectorAll('[href*="streamlit.io"], [href*="streamlit.app"]');
    sel.forEach(e => e.style.display='none');
} catch(e) { console.warn('parent DOM not reachable', e); }
</script>
""", height=0)

# --- Page Configuration ---
st.set_page_config(page_title="CSV Visualizer & Forecasting", page_icon="📊", layout="centered")

# --- Modern Pastel Theme CSS ---
page_style = """
<style>
body, .stApp {
    background: linear-gradient(180deg, #f3f9ff, #fdfcff);
    font-family: 'Poppins', sans-serif;
    color: #1a1a1a;
}

/* --- Title --- */
.main-title {
    text-align: center;
    font-size: 2.6rem;
    font-weight: 800;
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
    background: linear-gradient(90deg, #007BFF, #00C6A2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* --- Subtitle --- */
.subtitle {
    text-align: center;
    font-size: 1.1rem;
    color: #333;
    opacity: 0.9;
    margin-bottom: 2rem;
}

/* --- Card Container --- */
.card {
    background: rgba(255, 255, 255, 0.7);
    border-radius: 25px;
    box-shadow: 0 6px 15px rgba(0,0,0,0.08);
    padding: 2rem;
    margin: 1.5rem auto;
    width: 90%;
    max-width: 700px;
    text-align: center;
    backdrop-filter: blur(15px);
    border: 1px solid rgba(255,255,255,0.6);
    transition: transform 0.25s ease, box-shadow 0.3s ease;
}
.card:hover {
    transform: scale(1.02);
    box-shadow: 0 10px 25px rgba(0,0,0,0.15);
}

/* --- Card Headers with Pastel Gradients --- */
.card-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: white;
    border-radius: 14px;
    padding: 0.8rem 1rem;
    display: inline-block;
    margin-bottom: 1rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.card-header.forecast { background: linear-gradient(90deg, #a18cd1, #fbc2eb); }
.card-header.analysis { background: linear-gradient(90deg, #89f7fe, #66a6ff); }
.card-header.results { background: linear-gradient(90deg, #ffecd2, #fcb69f); }

/* --- Paragraphs --- */
.card p {
    font-size: 1rem;
    color: #222;
    line-height: 1.6;
    margin-bottom: 1.5rem;
}

/* --- File Upload Box --- */
[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed rgba(0,0,0,0.2);
    background: rgba(255,255,255,0.9);
    border-radius: 15px;
    transition: all 0.3s ease;
    color: #333 !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #66a6ff;
    background: rgba(255,255,255,1);
    box-shadow: 0 0 12px rgba(102,166,255,0.3);
}
[data-testid="stFileUploaderDropzone"] * {
    color: #333 !important;
}

/* --- Buttons --- */
.stButton > button {
    background: linear-gradient(90deg, #36D1DC, #5B86E5);
    border: none;
    color: white;
    font-weight: 600;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 10px rgba(91,134,229,0.3);
}
.stButton > button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #5B86E5, #36D1DC);
}

/* --- Hide Streamlit Defaults --- */
#MainMenu, footer, header, [data-testid="stToolbar"], [data-testid="stStatusWidget"] {
    display: none !important;
}
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

# Safe import for Streamlit script context across all Streamlit versions
try:
    from streamlit.runtime.scriptrunner_utils.script_run_context import add_script_run_ctx, get_script_run_ctx
except ImportError:
    try:
        from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
    except ImportError:
        from streamlit.scriptrunner import add_script_run_ctx, get_script_run_ctx

from crewai import Agent, Task, Crew, LLM
from crewai_tools import EXASearchTool, ScrapeWebsiteTool

# --- SUPPRESS THREAD WARNING LOGS ---
logging.getLogger("streamlit.runtime.scriptrunner.script_runner").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.state.session_state_proxy").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

# Fix asyncio event loop for Streamlit runner
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

st.title("🤖 AI Research Crew")
st.markdown(
    "Break down complex queries into deep research plans, gather web data, "
    "fact-check findings, and write actionable reports using **CrewAI** and **Gemini**."
)

# Retrieve keys from st.secrets if available, fallback to empty string
gemini_key = (st.secrets.get("GEMINI_API_KEY", "") if "GEMINI_API_KEY" in st.secrets else "")
exa_key = (st.secrets.get("EXA_API_KEY", "") if "EXA_API_KEY" in st.secrets else "")


# --- THREAD-SAFE STDOUT REDIRECTOR ---
class StreamlitLogRedirector(io.StringIO):
    """
    Thread-safe stream redirector that captures CrewAI terminal output 
    without breaking runtime listeners or raising ScriptRunContext errors.
    """
    def __init__(self, placeholder, ctx):
        super().__init__()
        self.placeholder = placeholder
        self.ctx = ctx
        self.buffer = ""

    def write(self, string):
        self.buffer += string
        clean_text = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', self.buffer)

        if get_script_run_ctx() is None and self.ctx is not None:
            try:
                add_script_run_ctx(ctx=self.ctx)
            except Exception:
                pass

        try:
            self.placeholder.code(clean_text, language="bash")
            st.session_state['execution_logs'] = clean_text
        except Exception:
            pass
        return len(string)

    def flush(self):
        pass

    def isatty(self):
        return False


# --- GUARDRAIL FUNCTION ---
def write_report_guardrail(output):
    try:
        raw_output = output if isinstance(output, str) else output.raw
    except Exception as e:
        return (False, f"Error retrieving output: {str(e)}")

    output_lower = raw_output.lower()

    if not re.search(r'#+.*summary', output_lower):
        return (False, "The report must include a Summary section with a header like '## Summary'")

    if not re.search(r'#+.*insights|#+.*recommendations', output_lower):
        return (False, "The report must include an Insights section with a header like '## Insights'")

    if not re.search(r'#+.*citations|#+.*references', output_lower):
        return (False, "The report must include a Citations (or References) section with a header like '## Citations'")

    return (True, raw_output)


# --- MAIN INPUT SECTION WITH SESSION STATE KEY ---
user_query = st.text_area(
    "🔍 Enter your Research Query / Prompt:",
    key="user_query_input",
    height=120
)
run_button = st.button("🚀 Run Research Crew", type="primary", use_container_width=True)

# --- CREW EXECUTION FLOW ---
if run_button:
    if not gemini_key or not exa_key:
        st.error("Please provide both **Gemini API Key** and **Exa API Key** in secrets or the sidebar to proceed.")
    elif not user_query.strip():
        st.warning("Please enter a valid research query.")
    else:
        # Clear previous state
        st.session_state.pop('report_txt', None)
        st.session_state.pop('execution_logs', None)

        os.environ["GEMINI_API_KEY"] = gemini_key
        os.environ["EXA_API_KEY"] = exa_key
        os.environ["CREWAI_TESTING"] = "true"
        os.environ["CREWAI_TRACING_ENABLED"] = "false"

        old_stdout = sys.stdout
        current_ctx = get_script_run_ctx()

        with st.status("🤖 **Research Crew is working...**", expanded=True) as status:
            try:
                st.write("📋 **Live Agent Execution Logs:**")
                log_expander = st.expander("Show/Hide Live Agent Thoughts", expanded=True)
                log_placeholder = log_expander.empty()

                redirector = StreamlitLogRedirector(log_placeholder, ctx=current_ctx)
                sys.stdout = redirector

                # 1. Tools
                st.write("🔧 Initializing Exa Search & Web Scraping tools...")
                exa_search_tool = EXASearchTool(api_key=exa_key)
                scrape_website_tool = ScrapeWebsiteTool()

                # 2. Gemini LLM
                gemini_llm = LLM(
                    model="gemini/gemini-3.5-flash-lite",
                    api_key=gemini_key,
                    temperature=0.7
                )

                # 3. Agents
                st.write("👥 Assembling research agents (Planner, Researcher, Fact Checker, Writer)...")
                research_planner = Agent(
                    role="Research Planner",
                    goal="Analyze queries and break them down into smaller, specific research topics.",
                    backstory="You are a research strategist who excels at breaking down complex questions.",
                    llm=gemini_llm,
                    verbose=True,
                    max_rpm=150,
                    max_iter=15
                )

                researcher = Agent(
                    role="Internet Researcher",
                    goal="Research thoroughly all assigned topics",
                    backstory="You are a skilled researcher with experience in online investigation.",
                    tools=[exa_search_tool, scrape_website_tool],
                    llm=gemini_llm,
                    verbose=True,
                    max_rpm=150,
                    max_iter=15
                )

                fact_checker = Agent(
                    role="Fact Checker",
                    goal="Verify data for accuracy, identify inconsistencies, and flag potential misinformation",
                    backstory="You are a quality assurance specialist with expertise in fact-checking.",
                    tools=[exa_search_tool, scrape_website_tool],
                    llm=gemini_llm,
                    verbose=True,
                    max_rpm=150,
                    max_iter=15
                )

                report_writer = Agent(
                    role="Report Writer",
                    goal="Write clear, concise, and well-structured reports with mandatory headers (Summary, Insights, Citations).",
                    backstory="You are an expert writer who specializes in creating clear, well-structured research reports.",
                    llm=gemini_llm,
                    verbose=True,
                    max_rpm=150,
                    max_iter=15
                )

                # 4. Tasks
                create_research_plan_task = Task(
                    description="Based on the user's query, break it down into specific topics: {user_query}",
                    expected_output="A research plan with main research topics and key questions.",
                    agent=research_planner,
                )

                gather_research_data_task = Task(
                    description="Collect detailed information on all identified topics with citations.",
                    expected_output="Comprehensive research data and source credibility notes.",
                    agent=researcher,
                )

                verify_information_quality_task = Task(
                    description="Review collected research and verify facts against potential misinformation.",
                    expected_output="A report with original data and verified facts.",
                    agent=fact_checker,
                )

                write_final_report_task = Task(
                    description="Create a report using verified data. Must contain '## Summary', '## Insights', and '## Citations'.",
                    expected_output="A final research report containing ## Summary, ## Insights, and ## Citations.",
                    agent=report_writer,
                    guardrail=write_report_guardrail
                )

                # 5. Build and Kickoff Crew
                crew = Crew(
                    agents=[research_planner, researcher, fact_checker, report_writer],
                    tasks=[
                        create_research_plan_task,
                        gather_research_data_task,
                        verify_information_quality_task,
                        write_final_report_task
                    ],
                    memory=False
                )

                result = crew.kickoff(inputs={"user_query": user_query})

                # 6. Safely Extract Final Text
                report_txt = ""
                if hasattr(result, 'raw') and result.raw:
                    report_txt = result.raw
                elif hasattr(result, 'tasks_output') and result.tasks_output:
                    for task_out in reversed(result.tasks_output):
                        if task_out and hasattr(task_out, 'raw') and task_out.raw:
                            report_txt = task_out.raw
                            break
                if not report_txt:
                    report_txt = str(result)

                # Save directly to file synchronously
                with open("research_report.txt", "w", encoding="utf-8") as f:
                    f.write(report_txt)

                # Store in session state for persistence
                st.session_state['report_txt'] = report_txt

                status.update(label="✅ **Research complete!**", state="complete", expanded=False)

            except Exception as e:
                status.update(label="❌ **An error occurred during execution.**", state="error", expanded=True)
                st.error(f"Error during execution: {str(e)}")
            finally:
                sys.stdout = old_stdout

# --- DISPLAY LOGS & OUTPUT SECTION (PERSISTENT OUTSIDE BUTTON SCOPE) ---
if 'execution_logs' in st.session_state:
    with st.expander("📋 Execution Logs", expanded=False):
        st.code(st.session_state['execution_logs'], language="bash")

if 'report_txt' in st.session_state:
    st.markdown("---")
    st.header("📄 Final Generated Research Report")

    st.download_button(
        label="📥 Download Report (.txt)",
        data=st.session_state['report_txt'],
        file_name="research_report.txt",
        mime="text/plain",
        use_container_width=True
    )

    st.markdown(st.session_state['report_txt'])
