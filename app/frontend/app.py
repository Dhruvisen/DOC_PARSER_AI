import streamlit as st
import requests
import pandas as pd
import json
import io

# --- CONFIGURATION ---
if "backend_url" not in st.session_state:
    st.session_state.backend_url = "http://127.0.0.1:8001"

st.set_page_config(
    page_title="DOC PARSER AI - Multi-Agent System",
    page_icon=None,
    layout="wide",
)

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_id" not in st.session_state:
    st.session_state.user_id = "Default User"
if "last_processed_files" not in st.session_state:
    st.session_state.last_processed_files = []
if "processed_filenames" not in st.session_state:
    st.session_state.processed_filenames = set()

# --- SIDEBAR ---
with st.sidebar:
    st.title("Configuration")
    st.session_state.backend_url = st.text_input("Backend URL", value=st.session_state.backend_url)
    BACKEND_URL = st.session_state.backend_url
    
    st.session_state.user_id = st.text_input("User ID", value=st.session_state.user_id)
    
    st.divider()
    if st.button("Clear RAG Index"):
        with st.spinner("Clearing index..."):
            try:
                response = requests.post(f"{BACKEND_URL}/clear-index")
                if response.status_code == 200:
                    st.success("Index cleared successfully!")
                    st.session_state.messages = []
                    st.session_state.last_processed_files = []
                    st.session_state.processed_filenames = set()
                else:
                    st.error(f"Failed to clear index: {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    st.divider()
    st.subheader("Add New Documents")
    uploaded_files = st.file_uploader(
        "Choose files to analyze",
        accept_multiple_files=True,
        type=["pdf", "csv", "xlsx", "xls", "png", "jpg", "jpeg", "zip"],
        label_visibility="collapsed"
    )
    
    # Filter for new files only
    new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_filenames] if uploaded_files else []
    
    if st.button("Upload and Analyze", disabled=not new_files, use_container_width=True):
        with st.spinner(f"Reading {len(new_files)} files..."):
            files = [("files", (f.name, f.getvalue())) for f in new_files]
            data = {"user_id": st.session_state.user_id}
            
            try:
                response = requests.post(f"{BACKEND_URL}/parse", files=files, data=data)
                if response.status_code == 200:
                    result = response.json()
                    new_results = result.get("data", [])
                    
                    # Append new results to history
                    st.session_state.last_processed_files.extend(new_results)
                    
                    # Mark as processed
                    for f in new_files:
                        st.session_state.processed_filenames.add(f.name)
                        
                    st.success(f"Successfully processed {len(new_files)} new files!")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Connection failed: {str(e)}")

    if st.session_state.processed_filenames:
        st.caption(f"Processed: {', '.join(st.session_state.processed_filenames)}")

# --- MAIN PAGE ---
st.title("DOC PARSER AI")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Chat (RAG)", "Data Analysis", "Report Studio"])

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
    
    # Show everything in history to avoid "missing file" confusion
    history = st.session_state.last_processed_files
    
    if not history:
        st.info("No files have been analyzed yet. Please upload a CSV or Excel file and click 'Upload and Analyze'.")
    else:
        # Show all files, newest first
        history_reversed = history[::-1]
        
        file_options = {}
        for idx, f in enumerate(history_reversed):
            status_prefix = "" if f.get("status") == "success" else "[FAILED] "
            label = f"{status_prefix}{f['filename']} (Ref: {f.get('data', {}).get('doc_id', 'N/A')[:6]})"
            
            # Ensure unique keys for selectbox
            unique_label = f"{idx+1}. {label}"
            file_options[unique_label] = f
            
        selected_label = st.selectbox("Select a file from history", options=list(file_options.keys()))
        f = file_options[selected_label]    
        
        st.divider()
        
        if f.get("status") != "success":
            st.error(f"Error processing this file: {f.get('error', 'Unknown Error')}")
        else:
            file_data = f.get("data", {})
            file_type = file_data.get("file_type")
            
            st.subheader(f"Analysis for: {f['filename']}")
            
            if file_type not in ["csv", "excel"]:
                st.warning(f"This file type ({file_type}) does not support deep data analysis. Please upload a CSV or Excel file.")
            else:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("Automatic Insights")
                    insights = file_data.get("data_analysis")
                    if insights:
                        st.markdown(insights)
                        if st.button("🔄 Refresh Insights", key=f"re_auto_{file_data.get('doc_id')}"):
                            with st.spinner("Updating insights..."):
                                # We can't easily re-upload, but we can call analyze with the generic question
                                # Since we don't have bytes here, we notify the user or we'd need to fetch from MinIO
                                st.info("To fully regenerate, use the 'Deep Dive' with: 'Provide a general high-level summary'")
                    else:
                        st.write("No automatic insights generated.")
                        if st.button("⚡ Generate Now", key=f"re_auto_{file_data.get('doc_id')}"):
                            # Use the deep dive logic but for the automatic section
                            with st.spinner("Analyzing dataset..."):
                                data = {
                                    "doc_id": file_data.get("doc_id"),
                                    "user_id": st.session_state.user_id,
                                    "question": "Provide a general high-level summary and identify any interesting trends or outliers in this dataset."
                                }
                                res = requests.post(f"{BACKEND_URL}/analyze-data", data=data)
                                if res.status_code == 200:
                                    ans = res.json()
                                    if ans.get("status") == "success":
                                        # Update session state history
                                        f["data"]["data_analysis"] = ans.get("insight")
                                        st.success("Insights Generated!")
                                        st.rerun()
                                    else:
                                        st.error("Failed to generate insights automatically.")
                                else:
                                    st.error("Error connecting to analyzer.")
                
                with col2:
                    st.subheader("Deep Dive Analysis")
                    q = st.text_input("Ask a specific data question", key=f"q_{file_data.get('doc_id')}")
                    if st.button("Run Python Analysis", key=f"btn_{file_data.get('doc_id')}"):
                        with st.spinner("Executing pandas code..."):
                            data = {
                                "doc_id": file_data.get("doc_id"),
                                "user_id": st.session_state.user_id,
                                "question": q
                            }
                            try:
                                res = requests.post(f"{BACKEND_URL}/analyze-data", data=data)
                                if res.status_code == 200:
                                    analysis = res.json()
                                    if analysis.get("status") == "success":
                                        st.success("Analysis Complete!")
                                        
                                        # Show the human-readable insight
                                        st.subheader("Analysis Finding")
                                        st.markdown(analysis.get("insight"))
                                        
                                        # Show the raw result data in a clean way
                                        with st.expander("View Raw Result Data", expanded=True):
                                            res_val = analysis.get("data_preview")
                                            if res_val:
                                                st.text(res_val)
                                        
                                        # Hide the Python code for non-technical users
                                        with st.expander("🛠️ View Technical Details (Python Code)"):
                                            st.code(analysis.get("generated_code"), language="python")
                                    else:
                                        st.error(analysis.get("error", "Analysis logic failed."))
                                else:
                                    st.error(f"Backend error: {res.status_code}")
                            except Exception as e:
                                st.error(f"Connection failed: {str(e)}")

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
                    
                    st.subheader("Email Draft")
                    st.text_area("Copy this email:", value=reports.get("email_draft", ""), height=200)
                    
                    st.subheader("Management Summary")
                    st.markdown(reports.get("management_summary", ""))
                    
                    st.subheader("Bullet Report")
                    st.markdown(reports.get("bullet_report", ""))
                else:
                    st.error("Failed to generate report.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
