from __future__ import annotations

import contextlib
import io
import json
from datetime import date, datetime
from pathlib import Path

import streamlit as st

# NOTE: adjust this import if your pipeline file has a different name.
from bwa_backend import run

# ──────────────────────────────────────────
# Paths
# ──────────────────────────────────────────
OUTPUT_DIR = Path("generated_blogs")          # where this app stores its own run records
IMAGES_DIR = Path("images")                    # matches blog_writer.py's image output folder
OUTPUT_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Blog Writing Agent", layout="wide", page_icon="📝")

# ──────────────────────────────────────────
# Styling — dark theme + red accent to match the mock
# (.streamlit/config.toml sets the base dark theme; this adds the finer touches)
# ──────────────────────────────────────────
st.markdown(
    """
    <style>
    div[data-testid="stTextInput"] input,
    div[data-testid="stDateInput"] input {
        border: 1px solid #ff4b4b !important;
        border-radius: 6px;
    }

    div[data-testid="stButton"] button {
        background-color: #ff4b4b;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
        padding: 0.6rem 0;
    }
    div[data-testid="stButton"] button:hover {
        background-color: #ff6b6b;
        color: white;
    }

    /* red radio dot for the "Past blogs" list */
    div[role="radiogroup"] label span:first-child {
        border-color: #ff4b4b !important;
    }

    section[data-testid="stSidebar"] hr {
        margin: 0.75rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────

def _slug(text: str) -> str:
    keep = "".join(c if c.isalnum() or c in (" ", "-", "_") else "" for c in text)
    return "_".join(keep.split()).strip("_") or "untitled"


def save_record(out: dict, topic: str, as_of: str, logs: str) -> Path:
    """Persist a run's full output so it can be reloaded later from 'Past blogs'."""
    plan = out.get("plan")
    evidence = out.get("evidence", [])
    record = {
        "topic": topic,
        "as_of": as_of,
        "mode": out.get("mode"),
        "plan": plan.model_dump() if plan else None,
        "evidence": [e.model_dump() if hasattr(e, "model_dump") else e for e in evidence],
        "image_specs": out.get("image_specs", []),
        "final_markdown": out.get("final", ""),
        "logs": logs,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    title = plan.blog_title if plan else topic
    path = OUTPUT_DIR / f"{_slug(title)}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def load_record(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def list_past_blogs() -> list[Path]:
    return sorted(OUTPUT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def record_from_live_result(out: dict, logs: str) -> dict:
    """Shape a fresh run() output the same way a loaded JSON record looks."""
    plan = out.get("plan")
    return {
        "plan": plan.model_dump() if plan else None,
        "evidence": [e.model_dump() if hasattr(e, "model_dump") else e for e in out.get("evidence", [])],
        "image_specs": out.get("image_specs", []),
        "final_markdown": out.get("final", ""),
        "logs": logs,
    }


# ──────────────────────────────────────────
# Session state
# ──────────────────────────────────────────
if "live_result" not in st.session_state:
    st.session_state.live_result = None   # record dict for the blog just generated this session
if "active_blog" not in st.session_state:
    st.session_state.active_blog = None   # stem of a past blog selected from the sidebar


def _on_pick_past_blog():
    st.session_state.live_result = None
    st.session_state.active_blog = st.session_state.get("past_blog_choice")


# ──────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────
with st.sidebar:
    st.markdown("### Generate New Blog")

    topic = st.text_input("Topic", value="LightGBM")
    as_of = st.date_input("As-of date", value=date(2026, 1, 31))

    if st.button("🚀 Generate Blog"):
        log_buffer = io.StringIO()
        try:
            with st.spinner("Running the agent pipeline..."):
                with contextlib.redirect_stdout(log_buffer):
                    out = run(topic, as_of.isoformat())
            logs = log_buffer.getvalue()
            save_record(out, topic, as_of.isoformat(), logs)
            st.session_state.live_result = record_from_live_result(out, logs)
            st.session_state.active_blog = None
            st.success("Blog generated.")
        except Exception as e:
            st.session_state.live_result = {
                "plan": None, "evidence": [], "image_specs": [],
                "final_markdown": "", "logs": log_buffer.getvalue(),
            }
            st.error(f"Generation failed: {e}")

    st.markdown("---")
    st.markdown("### Past blogs")

    past = list_past_blogs()
    if not past:
        st.caption("No blogs generated yet.")
    else:
        label_map = {}
        for p in past:
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
                title = (rec.get("plan") or {}).get("blog_title") or p.stem
            except Exception:
                title = p.stem
            label_map[p.stem] = f"{title} · {p.stem}.md"

        stems = [p.stem for p in past]
        if st.session_state.active_blog is None and st.session_state.live_result is None:
            st.session_state.active_blog = stems[0]

        st.radio(
            "past_blogs",
            stems,
            format_func=lambda s: label_map.get(s, s),
            label_visibility="collapsed",
            key="past_blog_choice",
            on_change=_on_pick_past_blog,
        )

# ──────────────────────────────────────────
# Resolve what to display: a fresh run, or a selected past blog
# ──────────────────────────────────────────
display_record = None
if st.session_state.live_result is not None:
    display_record = st.session_state.live_result
elif st.session_state.active_blog:
    match = next((p for p in list_past_blogs() if p.stem == st.session_state.active_blog), None)
    if match:
        display_record = load_record(match)

# ──────────────────────────────────────────
# Main area
# ──────────────────────────────────────────
st.title("Blog Writing Agent")

tab_plan, tab_evidence, tab_preview, tab_images, tab_logs = st.tabs(
    ["🚀 Plan", "📄 Evidence", "📝 Markdown Preview", "🖼️ Images", "📋 Logs"]
)

with tab_plan:
    st.header("Plan")
    plan = (display_record or {}).get("plan")
    if not plan:
        st.info("Generate a blog or pick one from Past blogs to see its plan.")
    else:
        st.markdown(f"**Title:** {plan['blog_title']}")

        c1, c2, c3 = st.columns([3, 1, 1])
        c1.markdown(f"**Audience:** {plan['audience']}")
        c2.markdown(f"**Tone:** {plan['tone']}")
        c3.markdown(f"**Blog kind:** {plan['blog_kind']}")

        st.write("")

        rows = [
            {
                "id": t["id"],
                "title": t["title"],
                "target_words": t["target_words"],
                "requires_research": t["requires_research"],
                "requires_citations": t["requires_citations"],
                "requires_code": t["requires_code"],
                "tags": ", ".join(t.get("tags", [])),
            }
            for t in plan["tasks"]
        ]
        st.dataframe(
            rows,
            hide_index=True,
            use_container_width=True,
            column_config={
                "requires_research": st.column_config.CheckboxColumn("requires_research"),
                "requires_citations": st.column_config.CheckboxColumn("requires_citations"),
                "requires_code": st.column_config.CheckboxColumn("requires_code"),
            },
        )

with tab_evidence:
    st.header("Evidence")
    evidence = (display_record or {}).get("evidence") or []
    if not evidence:
        st.info("No evidence collected for this blog (closed-book mode, or nothing generated yet).")
    else:
        st.dataframe(evidence, hide_index=True, use_container_width=True)

with tab_preview:
    st.header("Markdown Preview")
    md = (display_record or {}).get("final_markdown") or ""
    if not md:
        st.info("No markdown yet.")
    else:
        st.markdown(md)

with tab_images:
    st.header("Images")
    specs = (display_record or {}).get("image_specs") or []
    if not specs:
        st.info("No images for this blog.")
    else:
        cols = st.columns(min(3, len(specs)))
        for i, spec in enumerate(specs):
            filename = Path(spec["filename"]).name
            img_path = IMAGES_DIR / filename
            with cols[i % len(cols)]:
                if img_path.exists():
                    st.image(str(img_path), caption=spec.get("caption", filename))
                else:
                    st.warning(f"Missing on disk: {filename}")
                st.caption(spec.get("alt", ""))

with tab_logs:
    st.header("Logs")
    logs = (display_record or {}).get("logs") or ""
    st.code(logs or "No logs yet.", language="text")