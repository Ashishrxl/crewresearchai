import os
import re
import warnings
import asyncio
import streamlit as st
from crewai import Agent, Task, Crew, LLM
from crewai_tools import EXASearchTool, ScrapeWebsiteTool

# Fix asyncio event loop for Streamlit runner
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Ignore warning messages
warnings.filterwarnings('ignore')

# --- STREAMLIT PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Research Crew",
    page_icon="🤖",
    layout="wide"
)


st.markdown(
    "Break down complex queries into deep research plans, gather web data, "
    "fact-check findings, and write actionable reports using **CrewAI** and **Gemini**."
)

# Retrieve keys from st.secrets if available
gemini_key = st.secrets.get("GEMINI_API_KEY", "") if "GEMINI_API_KEY" in st.secrets else ""
exa_key = st.secrets.get("EXA_API_KEY", "") if "EXA_API_KEY" in st.secrets else ""




# --- GUARDRAIL FUNCTION ---
def write_report_guardrail(output):
    """
    Validates that the final report contains required headers:
    - Summary
    - Insights/Recommendations
    - Citations/References
    """
    try:
        raw_output = output if isinstance(output, str) else output.raw
    except Exception as e:
        return (False, f"Error retrieving the `raw` argument: {str(e)}")

    output_lower = raw_output.lower()

    if not re.search(r'#+.*summary', output_lower):
        return (False, "The report must include a Summary section with a header like '## Summary'")

    if not re.search(r'#+.*insights|#+.*recommendations', output_lower):
        return (False, "The report must include an Insights section with a header like '## Insights'")

    if not re.search(r'#+.*citations|#+.*references', output_lower):
        return (False, "The report must include a Citations (or References) section with a header like '## Citations'")

    return (True, raw_output)


# --- SAVE FILE CALLBACK ---
def save_file_hook(result):
    """
    Saves the final report content cleanly to a local markdown file.
    """
    try:
        report_content = ""
        if hasattr(result, 'raw') and result.raw:
            report_content = result.raw
        elif hasattr(result, 'tasks_output') and result.tasks_output:
            for task_out in reversed(result.tasks_output):
                if task_out and hasattr(task_out, 'raw') and task_out.raw:
                    report_content = task_out.raw
                    break

        if not report_content:
            report_content = str(result) if result is not None else "No output generated."

        with open("research_report.md", "w", encoding="utf-8") as f:
            f.write(report_content)
    except Exception as e:
        st.error(f"Error saving report to file: {str(e)}")


# --- MAIN INPUT SECTION ---

user_query = st.text_area("🔍 Enter your Research Query / Prompt:",  height=120)

run_button = st.button("🚀 Run Research Crew", type="primary", use_container_width=True)

# --- CREW EXECUTION FLOW ---
if run_button:
    if not gemini_key or not exa_key:
        st.error("Please provide both **Gemini API Key** and **Exa API Key** in the sidebar to proceed.")
    elif not user_query.strip():
        st.warning("Please enter a valid research query.")
    else:
        # Set environment variables required by tools and CrewAI
        os.environ["CREWAI_TESTING"] = "true"
        os.environ["CREWAI_TRACING_ENABLED"] = "true"
        os.environ["GEMINI_API_KEY"] = gemini_key
        os.environ["EXA_API_KEY"] = exa_key

        with st.status("🤖 **Research Crew is working...**", expanded=True) as status:
            try:
                # 1. Initialize Tools
                st.write("🔧 Initializing Exa Search & Web Scraping tools...")
                exa_search_tool = EXASearchTool(base_url="https://api.exa.ai")
                scrape_website_tool = ScrapeWebsiteTool()

                # 2. Initialize Gemini Model
                gemini_llm = LLM(
                    model="gemini/gemini-1.5-flash",
                    api_key=gemini_key
                )

                # 3. Initialize Agents
                st.write("👥 Assembling research agents (Planner, Researcher, Fact Checker, Writer)...")
                research_planner = Agent(
                    role="Research Planner",
                    goal="Analyze queries and break them down into smaller, specific research topics.",
                    backstory=(
                        "You are a research strategist who excels at breaking down complex questions "
                        "into manageable research components. You identify what needs to be researched "
                        "and create clear research objectives."
                    ),
                    llm=gemini_llm,
                    verbose=True,
                    max_rpm=150,
                    max_iter=15
                )

                researcher = Agent(
                    role="Internet Researcher",
                    goal="Research thoroughly all assigned topics",
                    backstory=(
                        "You are a skilled researcher with experience in online investigation "
                        "and data collection. You know how to find reliable sources, extract relevant information, "
                        "and always verify facts across multiple sources to avoid misinformation or hallucination. "
                        "You never invent facts and always trace information to its origin."
                    ),
                    tools=[exa_search_tool, scrape_website_tool],
                    llm=gemini_llm,
                    verbose=True,
                    max_rpm=150,
                    max_iter=15
                )

                fact_checker = Agent(
                    role="Fact Checker",
                    goal="Verify data for accuracy, identify inconsistencies, and flag potential misinformation",
                    backstory=(
                        "You are a quality assurance specialist with expertise in fact-checking "
                        "and identifying misinformation and hallucinations. You cross-reference information, "
                        "spot inconsistencies, and ensure all data meets high accuracy standards. You rigorously "
                        "check for hallucinated or invented content and require that all facts be supported by evidence."
                    ),
                    tools=[exa_search_tool, scrape_website_tool],
                    llm=gemini_llm,
                    verbose=True,
                    max_rpm=150,
                    max_iter=15
                )

                report_writer = Agent(
                    role="Report Writer",
                    goal="Write clear, concise, and well-structured reports based on gathered information",
                    backstory=(
                        "You are an expert writer who specializes in creating clear, well-structured "
                        "research reports. You synthesize complex information into readable formats and "
                        "always include proper citations and sources."
                    ),
                    llm=gemini_llm,
                    verbose=True,
                    max_rpm=150,
                    max_iter=15
                )

                # 4. Initialize Tasks
                st.write("📋 Assigning tasks and applying guardrails...")
                create_research_plan_task = Task(
                    description=(
                        "Based on the user's query, break it down into specific topics and key questions, "
                        "and create a focused research plan. The user's query is: {user_query}"
                    ),
                    expected_output=(
                        "A research plan with main research topics to investigate, "
                        "key questions for each topic, and success criteria for the research."
                    ),
                    agent=research_planner,
                )

                gather_research_data_task = Task(
                    description=(
                        "Using the research plan, collect information on all identified topics. "
                        "Cite all sources used."
                    ),
                    expected_output=(
                        "Comprehensive research data including: information for each "
                        "research topic, and citations used along with source credibility notes"
                    ),
                    agent=researcher,
                )

                verify_information_quality_task = Task(
                    description=(
                        "Review all collected research. Identify any conflicting information, "
                        "potential misinformation, or gaps that need addressing."
                    ),
                    expected_output=(
                        "A report with all the original data you got plus any "
                        "verified facts vs. questionable information, make sure this is as comprehensive "
                        "as possible for final report generation"
                    ),
                    agent=fact_checker,
                )

                write_final_report_task = Task(
                    description=(
                        "Create a comprehensive report that answers the original query using all verified research data. "
                        "Structure it with clear sections, include citations, and provide actionable insights."
                    ),
                    expected_output=(
                        "A final research report containing: executive summary, detailed "
                        "findings that answer the user query, supporting evidence and analysis, complete "
                        "source citations."
                    ),
                    agent=report_writer,
                    guardrail=write_report_guardrail
                )

                # 5. Build and Run Crew
                st.write("🚀 Executing multi-agent collaboration workflow...")
                crew = Crew(
                    agents=[research_planner, researcher, fact_checker, report_writer],
                    tasks=[
                        create_research_plan_task,
                        gather_research_data_task,
                        verify_information_quality_task,
                        write_final_report_task
                    ],
                    memory=True,
                    after_kickoff_callbacks=[save_file_hook]
                )

                result = crew.kickoff(inputs={"user_query": user_query})

                status.update(label="✅ **Research complete!**", state="complete", expanded=False)

                # 6. Extract Output Content
                report_md = result.raw if hasattr(result, 'raw') and result.raw else str(result)

                # Store in session state for persistence
                st.session_state['report_md'] = report_md

            except Exception as e:
                status.update(label="❌ **An error occurred during execution.**", state="error", expanded=True)
                st.error(f"Error during execution: {str(e)}")

# --- DISPLAY OUTPUT SECTION ---
if 'report_md' in st.session_state:
    st.markdown("---")
    st.header("📄 Final Generated Research Report")

    # Download Button
    st.download_button(
        label="📥 Download Report (.md)",
        data=st.session_state['report_md'],
        file_name="research_report.md",
        mime="text/markdown",
        use_container_width=True
    )

    # Render Report Content
    st.markdown(st.session_state['report_md'])
