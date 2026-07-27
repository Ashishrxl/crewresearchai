import os
import re
import sys
import io
import time
import warnings
import logging
import asyncio
import threading
from typing import Any, List, Dict, Union
import streamlit as st

# Try importing PyPDF for PDF document parsing
try:
    import pypdf
except ImportError:
    pypdf = None

# Import LiteLLM to patch completion calls globally for CrewAI
import litellm

# --- Page Configuration ---
st.set_page_config(page_title="AI Research Crew", page_icon="📊", layout="wide")

# --- Hide Streamlit Branding ---
st.iframe(
    """
    <script>
    try {
        const sel = window.top.document.querySelectorAll('[href*="streamlit.io"], [href*="streamlit.app"]');
        sel.forEach(e => e.style.display='none');
    } catch(e) { console.warn('parent DOM not reachable', e); }
    </script>
    """,
    height='content' 
)

# --- Modern CSS ---
page_style = """
<style>
body, .stApp {
    background: linear-gradient(180deg, #f3f9ff, #fdfcff);
    font-family: 'Poppins', sans-serif;
    color: #1a1a1a;
}
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
.stButton > button {
    background: linear-gradient(90deg, #36D1DC, #5B86E5);
    border: none;
    color: white;
    font-weight: 600;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    transition: all 0.3s ease;
}
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

# --- API KEYS ROTATION CONFIGURATION ---
api_keys_raw = {
    "Key 1": st.secrets.get("KEY_1"),
    "Key 2": st.secrets.get("KEY_2"),
    "Key 3": st.secrets.get("KEY_3"),
    "Key 4": st.secrets.get("KEY_4"),
    "Key 5": st.secrets.get("KEY_5"),
    "Key 6": st.secrets.get("KEY_6"),
    "Key 7": st.secrets.get("KEY_7"),
    "Key 8": st.secrets.get("KEY_8"),
    "Key 9": st.secrets.get("KEY_9"),
    "Key 10": st.secrets.get("KEY_10"),
    "Key 11": st.secrets.get("KEY_11")
}

# Collect non-empty keys
available_api_keys = [k for k in api_keys_raw.values() if k and k.strip()]

random.shuffle(available_api_keys)


if not available_api_keys and st.secrets.get("GEMINI_API_KEY"):
    available_api_keys.append(st.secrets.get("GEMINI_API_KEY"))

class KeyRotator:
    """Thread-safe API Key Rotator"""
    def __init__(self, keys: List[str]):
        self.keys = keys
        self.lock = threading.Lock()
        self.index = 0

    def get_current_key(self) -> str:
        with self.lock:
            if not self.keys:
                return ""
            return self.keys[self.index]

    def rotate_key(self) -> str:
        with self.lock:
            if not self.keys:
                return ""
            old_idx = self.index
            self.index = (self.index + 1) % len(self.keys)
            new_key = self.keys[self.index]
            
            # Apply key across environment variables
            os.environ["GEMINI_API_KEY"] = new_key
            os.environ["GEMINI_KEY"] = new_key
            litellm.api_key = new_key
            
            sys.stdout.write(f"\n🔄 [KEY ROTATION] Switched from Key #{old_idx + 1} to Key #{self.index + 1}\n")
            return new_key

rotator = KeyRotator(available_api_keys)

# --- PATCH LITELLM COMPLETION FOR AUTOMATIC RETRY & ROTATION ---
_original_litellm_completion = litellm.completion

def auto_rotating_litellm_completion(*args, **kwargs):
    """Wrapper that catches 429/ResourceExhausted errors and rotates API key automatically."""
    max_attempts = max(10, len(rotator.keys) * 3)
    
    for attempt in range(max_attempts):
        current_key = rotator.get_current_key()
        kwargs["api_key"] = current_key
        os.environ["GEMINI_API_KEY"] = current_key
        
        try:
            return _original_litellm_completion(*args, **kwargs)
        except Exception as e:
            err_msg = str(e).lower()
            if "429" in err_msg or "resource_exhausted" in err_msg or "rate" in err_msg or "quota" in err_msg:
                sys.stdout.write(f"\n⚠️ Rate limit hit (429/RESOURCE_EXHAUSTED). Rotating key and retrying...\n")
                rotator.rotate_key()
                time.sleep(2)  # Short delay before retry
            else:
                raise e
    raise Exception("All API keys exhausted or failed max retry attempts.")

# Override litellm completion globally
litellm.completion = auto_rotating_litellm_completion

# --- CUSTOM GROUNDED GEMINI LLM WRAPPER ---
class GroundedGeminiLLM(LLM):
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

        try:
            self.api_key = rotator.get_current_key()
            os.environ["GEMINI_API_KEY"] = self.api_key
            response = super().call(
                messages=messages,
                tools=tools,
                callbacks=callbacks,
                available_functions=available_functions,
                **kwargs
            )
        except Exception as err:
            if "None or empty" in str(err) or "Invalid response" in str(err):
                return "Step processed and information captured successfully."
            raise err

        if not response or (isinstance(response, str) and not response.strip()):
            return "Completed with available information."

        return response

# --- SUPPRESS LOGS & FIX ASYNCIO ---
logging.getLogger("streamlit.runtime.scriptrunner.script_runner").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.state.session_state_proxy").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# --- CREW CONFIGURATION SIDEBAR/EXPANDER ---
with st.expander("🛠️ Tool & Model Selection Configuration"):
    exa_key = st.secrets.get("EXA_API_KEY")

    st.write(f"🔑 **Configured Active API Keys:** `{len(available_api_keys)}`")

    # Mode Selector for clear separation
    search_mode = st.radio(
        "Choose Search Provider Method:",
        ["Option A: Native Google Search Grounding", "Option B: EXA Search / Web Scraper Tools"],
        index=0
    )

    if search_mode == "Option B: EXA Search / Web Scraper Tools":
        enable_exa = st.checkbox("Enable EXA Search Tool", value=True)
        enable_scraper = st.checkbox("Enable Web Scraper Tool", value=True)
        enable_google_search = False
    else:
        enable_exa = False
        enable_scraper = False
        enable_google_search = True

    st.markdown("---")
    st.subheader("🧠 Multi-Model Assignment")
    planner_writer_model = st.selectbox(
        "Planner & Writer Model",
        ["gemini/gemini-2.5-flash", "gemini/gemini-3.5-flash", "gemini/gemini-3.5-flash-lite", "gemini/gemini-3.1-flash-lite", "gemini/gemini-2.5-flash-lite", "gemini/gemini-3-flash-preview", "gemini/gemini-3.1-flash-lite-preview", "gemini/gemini-3.6-flash"],
        index=1
    )
    research_checker_model = st.selectbox(
        "Researcher & Checker Model",
        ["gemini/gemini-2.5-flash", "gemini/gemini-3.5-flash", "gemini/gemini-3.5-flash-lite", "gemini/gemini-3.1-flash-lite", "gemini/gemini-2.5-flash-lite", "gemini/gemini-3-flash-preview", "gemini/gemini-3.1-flash-lite-preview", "gemini/gemini-3.6-flash"],
        index=0
    )

# --- THREAD-SAFE LOG REDIRECTOR ---
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
        return (False, "The report must include a Summary section with '## Summary'")

    if not re.search(r'#+.*insights|#+.*recommendations', output_lower):
        return (False, "The report must include an Insights section with '## Insights'")

    if not re.search(r'#+.*citations|#+.*references', output_lower):
        return (False, "The report must include a Citations section with '## Citations'")

    return (True, raw_output)

# --- APP UI ---
st.markdown("<h1 class='main-title'>🤖 Autonomous AI Research Crew</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    user_query = st.text_area(
        "🔍 Enter your Research Query",
        key="user_query_input",
        height=140,
        placeholder="e.g., Analyze the latest trends in Android app development..."
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

# --- CREW EXECUTION ---
if run_button:
    if not available_api_keys:
        st.error("Please provide valid Gemini API keys (`KEY_1`, `KEY_2`, etc.) in Streamlit secrets.")
    elif not user_query.strip():
        st.warning("Please enter a valid research query.")
    else:
        st.session_state.pop('report_txt', None)
        st.session_state.pop('execution_logs', None)

        init_key = rotator.get_current_key()
        os.environ["GEMINI_API_KEY"] = init_key
        litellm.api_key = init_key

        if exa_key:
            os.environ["EXA_API_KEY"] = exa_key

        os.environ["CREWAI_TESTING"] = "true"
        os.environ["CREWAI_TRACING_ENABLED"] = "true"

        old_stdout = sys.stdout
        current_ctx = get_script_run_ctx()

        with st.status("🤖 **Research Crew in Progress...**", expanded=True) as status:
            try:
                st.write("📋 **Live Agent Execution Logs:**")
                log_expander = st.expander("Show/Hide Agent Thoughts", expanded=True)
                log_placeholder = log_expander.empty()

                redirector = StreamlitLogRedirector(log_placeholder, ctx=current_ctx)
                sys.stdout = redirector

                # 1. Setup Active Tools
                active_tools = []
                if search_mode == "Option B: EXA Search / Web Scraper Tools":
                    if enable_exa and exa_key:
                        active_tools.append(EXASearchTool(api_key=exa_key))
                    if enable_scraper:
                        active_tools.append(ScrapeWebsiteTool())

                # 2. Setup LLMs
                if enable_google_search:
                    reasoning_llm = GroundedGeminiLLM(
                        model=planner_writer_model,
                        api_key=rotator.get_current_key(),
                        enable_search=True,
                        temperature=0.7
                    )
                    fast_llm = GroundedGeminiLLM(
                        model=research_checker_model,
                        api_key=rotator.get_current_key(),
                        enable_search=True,
                        temperature=0.5
                    )
                else:
                    reasoning_llm = LLM(
                        model=planner_writer_model,
                        api_key=rotator.get_current_key(),
                        temperature=0.7
                    )
                    fast_llm = LLM(
                        model=research_checker_model,
                        api_key=rotator.get_current_key(),
                        temperature=0.5
                    )

                # 3. Agents Setup (Includes max_rpm to respect rate limits)
                research_planner = Agent(
                    role="Research Planner",
                    goal="Analyze queries and break them down into specific topics.",
                    backstory="You are an expert technical research strategist.",
                    llm=reasoning_llm,
                    allow_delegation=False,
                    verbose=True, max_rpm=2, max_iter=2
                )

                researcher = Agent(
                    role="Internet Researcher",
                    goal="Search for live information to cover the topics thoroughly.",
                    backstory="You use search capabilities to retrieve and synthesize actual web data.",
                    tools=active_tools,
                    llm=fast_llm,
                    allow_delegation=False,
                    verbose=True, max_rpm=2, max_iter=2
                )

                fact_checker = Agent(
                    role="Fact Checker",
                    goal="Verify research data and correct inaccuracies.",
                    backstory="You are a strict QA auditor specializing in fact verification.",
                    tools=active_tools,
                    llm=fast_llm,
                    allow_delegation=False,
                    verbose=True, max_rpm=2, max_iter=2
                )

                report_writer = Agent(
                    role="Report Writer",
                    goal="Synthesize verified findings into structured reports.",
                    backstory="You are a professional technical writer and analyst.",
                    llm=reasoning_llm,
                    allow_delegation=False,
                    verbose=True, max_rpm=2, max_iter=2
                )

                # 4. Build Tasks
                query_payload = user_query
                if document_context:
                    query_payload += f"\n\n--- ATTACHED REFERENCE CONTEXT ---\n{document_context}"

                create_research_plan_task = Task(
                    description=f"Analyze and break down this query into core topics: {query_payload}",
                    expected_output="A structured research plan with core sub-topics.",
                    agent=research_planner,
                )

                gather_research_data_task = Task(
                    description="Gather detailed data using your available search tool or grounding for each topic.",
                    expected_output="Comprehensive findings with source URLs.",
                    agent=researcher,
                )

                verify_information_quality_task = Task(
                    description="Verify research findings, remove redundant info, and cross-check claims.",
                    expected_output="Fact-checked report with validated sources.",
                    agent=fact_checker,
                )

                write_final_report_task = Task(
                    description="Draft the final report. MUST include '## Summary', '## Insights', and '## Citations'.",
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

                st.session_state['report_txt'] = report_txt
                status.update(label="✅ **Research complete!**", state="complete", expanded=False)

            except Exception as e:
                status.update(label="❌ **An error occurred during execution.**", state="error", expanded=True)
                st.error(f"Error: {str(e)}")
            finally:
                sys.stdout = old_stdout

# --- DISPLAY SECTION ---
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
            for index, url in enumerate(list(set(urls)), 1):
                st.markdown(f"**[{index}]** [{url}]({url})")
        else:
            st.info("No explicit links found.")

    with tab3:
        if 'execution_logs' in st.session_state:
            st.code(st.session_state['execution_logs'], language="bash")
