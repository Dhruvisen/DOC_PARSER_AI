import streamlit as st
import requests
import pandas as pd
import json
import io

# --- CONFIGURATION ---
BACKEND_URL = "http://localhost:8001"

st.set_page_config(
    page_title="DOC PARSER AI - Multi-Agent System",
    page_icon="🤖",
    layout="wide",
)

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_id" not in st.session_state:
    st.session_state.user_id = "Default User"
if "last_processed_files" not in st.session_state:
    st.session_state.last_processed_files = []

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Configuration")
    st.session_state.user_id = st.text_input("User ID", value=st.session_state.user_id)
    
    st.divider()
    st.subheader("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, CSV, Excel, Image, or ZIP",
        accept_multiple_files=True,
        type=["pdf", "csv", "xlsx", "xls", "png", "jpg", "jpeg", "zip"]
    )
    
    if st.button("🚀 Parse & Index", disabled=not uploaded_files):
        with st.spinner("Processing documents..."):
            files = [("files", (f.name, f.getvalue())) for f in uploaded_files]
            data = {"user_id": st.session_state.user_id}
            
            try:
                response = requests.post(f"{BACKEND_URL}/parse", files=files, data=data)
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.last_processed_files = result.get("data", [])
                    st.success(f"Successfully processed {len(uploaded_files)} files!")
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Connection failed: {str(e)}")

# --- MAIN PAGE ---
st.title("🤖 DOC PARSER AI")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["💬 Chat (RAG)", "📊 Data Analysis", "📝 Report Studio"])

# TAB 1: CHAT / RAG
with tab1:
    st.header("Chat with your Documents")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    payload = {"question": prompt, "user_id": st.session_state.user_id}
                    response = requests.post(f"{BACKEND_URL}/ask-doc", json=payload)
                    if response.status_code == 200:
                        answer = response.json().get("answer", "No answer found.")
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    else:
                        st.error("Failed to get answer from RAG.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# TAB 2: DATA ANALYSIS
with tab2:
    st.header("Structured Data Insights")
    
    structured_files = [f for f in st.session_state.last_processed_files if f.get("data", {}).get("file_type") in ["csv", "excel"]]
    
    if not structured_files:
        st.info("Upload a CSV or Excel file to see data analysis.")
    else:
        for f in structured_files:
            with st.expander(f"Analysis for: {f['filename']}", expanded=True):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("Automatic Insights")
                    insights = f.get("data", {}).get("data_analysis")
                    if insights:
                        st.markdown(insights)
                    else:
                        st.write("No automatic insights generated.")
                
                with col2:
                    st.subheader("Deep Dive Analysis")
                    q = st.text_input("Ask a specific data question", key=f"q_{f['data']['doc_id']}")
                    if st.button("Run Python Analysis", key=f"btn_{f['data']['doc_id']}"):
                        with st.spinner("Executing pandas code..."):
                            data = {
                                "doc_id": f["data"]["doc_id"],
                                "user_id": st.session_state.user_id,
                                "question": q
                            }
                            res = requests.post(f"{BACKEND_URL}/analyze-data", data=data)
                            if res.status_code == 200:
                                analysis = res.json()
                                if analysis.get("status") == "success":
                                    st.success("Analysis Complete!")
                                    st.markdown(analysis.get("insight"))
                                    st.code(analysis.get("generated_code"), language="python")
                                else:
                                    st.error(analysis.get("error", "Unknown error"))
                            else:
                                st.error("Analysis failed.")

# TAB 3: REPORT STUDIO
with tab3:
    st.header("Professional Report Generation")
    
    st.markdown("Generate summaries, emails, and reports from the currently indexed context.")
    
    report_topic = st.text_area("What should the report focus on?", "Summarize the key findings and trends from all uploaded documents.")
    
    if st.button("Generate Executive Reports"):
        with st.spinner("Writing reports..."):
            try:
                payload = {"question": report_topic, "user_id": st.session_state.user_id}
                response = requests.post(f"{BACKEND_URL}/generate-report", json=payload)
                if response.status_code == 200:
                    reports = response.json().get("data", {})
                    
                    st.subheader("📧 Email Draft")
                    st.text_area("Copy this email:", value=reports.get("email_draft", ""), height=200)
                    
                    st.subheader("📋 Management Summary")
                    st.markdown(reports.get("management_summary", ""))
                    
                    st.subheader("📊 Bullet Report")
                    st.markdown(reports.get("bullet_report", ""))
                else:
                    st.error("Failed to generate report.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
