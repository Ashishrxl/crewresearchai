import os
import re
import sys
import io
import warnings
import logging
import asyncio
from typing import Any, List, Dict, Union
import streamlit as st

# Try importing PyPDF for PDF document parsing
try:
    import pypdf
except ImportError:
    pypdf = None

# --- Page Configuration ---
st.set_page_config(page_title="AI Research Crew", page_icon="📊", layout="wide")

# --- Hide Streamlit Branding (Updated using st.iframe) ---
st.iframe(
    """
    <script>
    try {
        const sel = window.top.document.querySelectorAll('[href*="streamlit.io"], [href*="streamlit.app"]');
        sel.forEach(e => e.style.display='none');
    } catch(e) { console.warn('parent DOM not reachable', e); }
    </script>
    """,
    height=content
)

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
    margin-top: 1rem;
    margin-bottom: 0.5rem;
    background: linear-gradient(90deg, #007BFF, #00C6A2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* --- Card Container --- */
.card {
    background: rgba(255, 255, 255, 0.7);
    border-radius: 20px;
    box-shadow: 0 6px 15px rgba(0,0,0,0.08);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(15px);
    border: 1px solid rgba(255,255,255,0.6);
}

/* --- File Upload Box --- */
[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed rgba(0,0,0,0.2);
    background: rgba(255,255,255,0.9);
    border-radius: 15px;
    transition: all 0.3s ease;
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
    transform: scale(1.02);
    background: linear-gradient(90deg, #5B86E5, #36D1DC);
}

/* --- Hide Streamlit Defaults --- */
#MainMenu, footer, header, [data-testid="stToolbar"], [data-testid="stStatusWidget"] {
    display: none !important;
}
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

# Safe import for Streamlit script context
try:
    from streamlit.runtime.scriptrunner_utils.script_run_context import add_script_run_ctx, get_script_run_ctx
except ImportError:
    try:
        from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
    except ImportError:
        from streamlit.scriptrunner import add_script_run_ctx, get_script_run_ctx

from crewai import Agent, Task, Crew, LLM
from crewai_tools import EXASearchTool, ScrapeWebsiteTool

# --- CUSTOM LLM WITH SAFE RESPONSE HANDLING ---
class SafeGroundedGeminiLLM(LLM):
    def __init__(self, model: str, api_key: str, enable_search: bool = True, **kwargs):
        super().__init__(model=model, api_key=api_key, **kwargs)
        self.enable_search = enable_search

    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: List[Dict] | None = None,
        callbacks: List[Any] | None = None,
        available_functions: Dict[str, Any] | None = None,
        **kwargs,
    ) -> Union[str, Any]:
        
        if tools is None:
            tools = []

        if self.enable_search:
            search_tool_exists = any("googleSearch" in t or "google_search" in t for t in tools)
            if not search_tool_exists:
                tools.insert(0, {"googleSearch": {}})

        response = super().call(
            messages=messages,
            tools=tools,
            callbacks=callbacks,
            available_functions=available_functions,
            **kwargs
        )
        
        # Fallback to prevent None or empty response errors
        if not response or (isinstance(response, str) and not response.strip()):
            return "Task completed. Unable to retrieve additional details."
            
        return response

# --- SUPPRESS THREAD WARNING LOGS ---
logging.getLogger("streamlit.runtime.scriptrunner.script_runner").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.state.session_state_proxy").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

# Fix asyncio event loop for Streamlit runner
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


with st.expander("Crew configuration"):
    # API Keys Configuration
    gemini_key = st.secrets.get("GEMINI_API_KEY")
    exa_key = st.secrets.get("EXA_API_KEY")

    st.markdown("---")
    st.subheader("🛠️ Active Tools")
    enable_exa = st.checkbox("Enable Exa Search Tool", value=True)
    enable_scraper = st.checkbox("Enable Web Scraper Tool", value=True)

    st.markdown("---")
    st.subheader("🌐 Native Gemini Grounding")
    enable_google_search = st.checkbox("Google Search Grounding", value=True)

    st.markdown("---")
    st.subheader("🧠 Multi-Model Assignment")
    planner_writer_model = st.selectbox(
        "Planner & Writer Model",
        ["gemini/gemini-2.5-pro", "gemini/gemini-3.5-flash", "gemini/gemini-3.1-flash-lite", "gemini/gemini-3.5-flash-lite"],
        index=1
    )
    research_checker_model = st.selectbox(
        "Researcher & Checker Model",
        ["gemini/gemini-3.1-flash-lite", "gemini/gemini-3.5-flash", "gemini/gemini-3.5-flash-lite"],
        index=0
    )

# --- THREAD-SAFE STDOUT REDIRECTOR ---
class StreamlitLogRedirector(io.StringIO):
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

# --- APP UI ---
st.markdown("<h1 class='main-title'>🤖 Autonomous AI Research Crew</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Deconstruct complex questions, synthesize multi-source web data, and generate grounded reports.</p>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    user_query = st.text_area(
        "🔍 Enter your Research Query",
        key="user_query_input",
        height=140,
        placeholder="e.g., Analyze the performance and memory overhead of Jetpack Compose vs Traditional Views in Android 12+..."
    )

with col2:
    uploaded_file = st.file_uploader("📄 Attach Reference Document (Optional)", type=["txt", "md", "pdf"])
    document_context = ""
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.pdf'):
                if pypdf is not None:
                    pdf_reader = pypdf.PdfReader(uploaded_file)
                    for page in pdf_reader.pages:
                        document_context += page.extract_text() or ""
                else:
                    st.error("pypdf is required to read PDF files.")
            else:
                document_context = uploaded_file.read().decode("utf-8")
            
            if document_context:
                st.success(f"Attached: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error reading file: {e}")

run_button = st.button("🚀 Run Research Crew", type="primary", use_container_width=True)

# --- CREW EXECUTION FLOW ---
if run_button:
    if not gemini_key:
        st.error("Please provide a valid **GEMINI_API_KEY** in your Streamlit secrets.")
    elif not user_query.strip():
        st.warning("Please enter a valid research query.")
    else:
        st.session_state.pop('report_txt', None)
        st.session_state.pop('execution_logs', None)

        os.environ["GEMINI_API_KEY"] = gemini_key
        if exa_key:
            os.environ["EXA_API_KEY"] = exa_key

        os.environ["CREWAI_TESTING"] = "true"
        os.environ["CREWAI_TRACING_ENABLED"] = "false"

        old_stdout = sys.stdout
        current_ctx = get_script_run_ctx()

        with st.status("🤖 **Research Crew in Progress...**", expanded=True) as status:
            try:
                st.write("📋 **Live Agent Execution Logs:**")
                log_expander = st.expander("Show/Hide Agent Thoughts", expanded=True)
                log_placeholder = log_expander.empty()

                redirector = StreamlitLogRedirector(log_placeholder, ctx=current_ctx)
                sys.stdout = redirector

                # 1. Setup Custom CrewAI Tools
                active_tools = []
                if enable_exa and exa_key:
                    active_tools.append(EXASearchTool(api_key=exa_key))
                if enable_scraper:
                    active_tools.append(ScrapeWebsiteTool())

                # 2. Setup Native Gemini Grounding via Safe Class
                reasoning_llm = SafeGroundedGeminiLLM(
                    model=planner_writer_model,
                    api_key=gemini_key,
                    enable_search=enable_google_search,
                    temperature=0.7
                )

                fast_llm = SafeGroundedGeminiLLM(
                    model=research_checker_model,
                    api_key=gemini_key,
                    enable_search=enable_google_search,
                    temperature=0.5
                )

                # 3. Agents Initialization
                research_planner = Agent(
                    role="Research Planner",
                    goal="Analyze queries and break them down into specific, structured topics.",
                    backstory="You are an expert technical research strategist.",
                    llm=reasoning_llm,
                    verbose=True,
                    max_rpm=150,
                    max_iter=15
                )

                researcher = Agent(
                    role="Internet Researcher",
                    goal="Research assigned topics thoroughly using internet tools and grounding.",
                    backstory="You are an expert online investigator with deep analytical skills.",
                    tools=active_tools,
                    llm=fast_llm,
                    verbose=True,
                    max_rpm=150,
                    max_iter=15
                )

                fact_checker = Agent(
                    role="Fact Checker",
                    goal="Verify research data, check source credibility, and correct inaccuracies.",
                    backstory="You are a strict QA auditor specializing in technical fact verification.",
                    tools=active_tools,
                    llm=fast_llm,
                    verbose=True,
                    max_rpm=150,
                    max_iter=15
                )

                report_writer = Agent(
                    role="Report Writer",
                    goal="Synthesize verified findings into structured reports (## Summary, ## Insights, ## Citations).",
                    backstory="You are a professional technical writer and analyst.",
                    llm=reasoning_llm,
                    verbose=True,
                    max_rpm=150,
                    max_iter=15
                )

                # 4. Build Tasks Context
                query_payload = user_query
                if document_context:
                    query_payload += f"\n\n--- ATTACHED REFERENCE CONTEXT ---\n{document_context}"

                create_research_plan_task = Task(
                    description=f"Analyze and break down this query into core topics: {query_payload}",
                    expected_output="A structured research plan with core sub-topics and key questions.",
                    agent=research_planner,
                )

                gather_research_data_task = Task(
                    description="Gather detailed data for each topic in the research plan.",
                    expected_output="Comprehensive findings with source URLs.",
                    agent=researcher,
                )

                verify_information_quality_task = Task(
                    description="Verify research findings, remove redundant info, and cross-check claims.",
                    expected_output="Fact-checked report with validated sources.",
                    agent=fact_checker,
                )

                write_final_report_task = Task(
                    description="Draft the final structured report. MUST include '## Summary', '## Insights', and '## Citations'.",
                    expected_output="A final markdown report containing ## Summary, ## Insights, and ## Citations.",
                    agent=report_writer,
                    guardrail=write_report_guardrail
                )

                # 5. Kickoff Crew
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

                result = crew.kickoff()

                # Extract Output safely
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

                with open("research_report.txt", "w", encoding="utf-8") as f:
                    f.write(report_txt)

                st.session_state['report_txt'] = report_txt
                status.update(label="✅ **Research complete!**", state="complete", expanded=False)

            except Exception as e:
                status.update(label="❌ **An error occurred during execution.**", state="error", expanded=True)
                st.error(f"Error: {str(e)}")
            finally:
                sys.stdout = old_stdout

# --- MULTI-TAB DISPLAY SECTION ---
if 'report_txt' in st.session_state:
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📄 Final Research Report", "🔗 Extracted Citations", "📋 Execution Logs"])

    with tab1:
        st.download_button(
            label="📥 Download Report (.txt)",
            data=st.session_state['report_txt'],
            file_name="research_report.txt",
            mime="text/plain",
            use_container_width=True
        )
        st.markdown(st.session_state['report_txt'])

    with tab2:
        st.subheader("Extracted References & Links")
        urls = re.findall(r'https?://[^\s\)]+', st.session_state['report_txt'])
        if urls:
            unique_urls = list(set(urls))
            for index, url in enumerate(unique_urls, 1):
                st.markdown(f"**[{index}]** [{url}]({url})")
        else:
            st.info("No explicit HTTP/HTTPS links extracted in the text.")

    with tab3:
        if 'execution_logs' in st.session_state:
            st.code(st.session_state['execution_logs'], language="bash")
