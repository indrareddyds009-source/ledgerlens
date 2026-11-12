"""Streamlit chat UI for LedgerLens. Run: streamlit run app/streamlit_app.py"""
import uuid

import streamlit as st

from ledgerlens.agent import ask, build_agent
from ledgerlens.audit import log_turn
from ledgerlens.data import CUSTOMERS, build_ledger

st.set_page_config(page_title="LedgerLens", page_icon="💳")


@st.cache_resource
def con():
    return build_ledger()


@st.cache_resource
def agent(cid):
    return build_agent(con(), cid)


st.sidebar.title("💳 LedgerLens")
cid = st.sidebar.selectbox("Signed in as (demo)", list(CUSTOMERS), format_func=lambda c: CUSTOMERS[c]["name"])
if st.session_state.get("cid") != cid:
    st.session_state.update(cid=cid, thread=uuid.uuid4().hex, chat=[])
st.sidebar.caption("Synthetic data, Jan–Jun 2026. Try:")
examples = ["How much did I spend in March?", "What about April?", "Top 5 merchants in Q1",
            "Compare January and May", "Any unusual transactions?", "Dining trend by month"]
picked = next((e for e in examples if st.sidebar.button(e)), None)

for role, text, trace in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(text)
        if trace:
            with st.expander("Tool calls"):
                st.json(trace)

q = st.chat_input("Ask about your transactions") or picked
if q:
    st.session_state.chat.append(("user", q, None))
    r = ask(agent(cid), q, thread_id=st.session_state.thread)
    log_turn(cid, st.session_state.thread, q, r)
    st.session_state.chat.append(("assistant", r["answer"], {"tool_calls": r["tool_calls"], "blocked": r["blocked"],
                                                            "latency_ms": r["latency_ms"]}))
    st.rerun()
