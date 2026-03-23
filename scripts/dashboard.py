"""Competition Analytics Dashboard.

Usage:
  uv run streamlit run scripts/dashboard.py
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

EVAL_FILE = "eval_results.json"
EXPERIMENTS_CSV = "experiments.csv"

# Leaderboard results (public test scores from competition)
LEADERBOARD = [
    {"submission": "yolov8x_onnx_clean", "public_test": 0.7802, "models": "1x YOLOv8x", "technique": "Single model baseline"},
    {"submission": "yolov8l_640_s77", "public_test": 0.7700, "models": "1x YOLOv8l", "technique": "Architecture variant"},
    {"submission": "onnx_ensemble_ms_tuned", "public_test": 0.9091, "models": "YOLOv8x+m+s", "technique": "WBF ensemble"},
    {"submission": "softvote_3x_fp16", "public_test": 0.9142, "models": "3x YOLOv8x", "technique": "Soft class voting"},
    {"submission": "wbf_3x_fp16_tta", "public_test": 0.9160, "models": "3x YOLOv8x", "technique": "WBF + TTA hflip"},
    {"submission": "wbf_2x1l_multiscale_tta", "public_test": 0.9210, "models": "2x YOLOv8x + 1x YOLOv8l", "technique": "Multi-scale + arch diversity"},
    {"submission": "sub1_triple_arch", "public_test": 0.9190, "models": "YOLOv8x + YOLO11x + YOLOv8l", "technique": "Triple arch (first attempt)"},
    {"submission": "candidate11_fulldata_noflip_l", "public_test": 0.9215, "models": "YOLOv8x + noflip + YOLOv8l", "technique": "Training diversity"},
    {"submission": "sub2_c11_yolo11x_swap (WINNER)", "public_test": 0.9226, "models": "YOLOv8x + YOLO11x + YOLOv8l", "technique": "Architecture diversity"},
]

FINAL_LEADERBOARD = [
    {"rank": 1, "team": "Experis", "private_test": 0.7113},
    {"rank": 2, "team": "sokfirma.no", "private_test": 0.7108},
    {"rank": 3, "team": "SingularIT", "private_test": 0.7096},
    {"rank": 4, "team": "Good vibez", "private_test": 0.7095},
    {"rank": 5, "team": "Løkka Language Models", "private_test": 0.7095},
    {"rank": 6, "team": "J6X", "private_test": 0.7094},
    {"rank": 7, "team": "Human-Like", "private_test": 0.7085},
    {"rank": 8, "team": "Morten Punnerud-Engelstad", "private_test": 0.7085},
    {"rank": 9, "team": "PH", "private_test": 0.7083},
    {"rank": 10, "team": "Norne", "private_test": 0.7081},
]


def main():
    st.set_page_config(layout="wide", page_title="NM i AI 2026 — Detection Analytics", page_icon="🏆")
    st.title("NM i AI 2026 — NorgesGruppen Detection Analytics")
    st.caption("Team Experis — 1st Place (0.7113 mAP private test)")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Final Leaderboard", "Submission Journey", "Training Runs", "Local Evaluation", "Technique Analysis"
    ])

    # === Tab 1: Final Leaderboard ===
    with tab1:
        st.header("Final Private Test Leaderboard")
        df_lb = pd.DataFrame(FINAL_LEADERBOARD)
        df_lb["margin"] = df_lb["private_test"] - df_lb["private_test"].iloc[0]
        df_lb.loc[0, "margin"] = 0.0

        col1, col2, col3 = st.columns(3)
        col1.metric("Our Score", "0.7113", "1st Place")
        col2.metric("Margin over 2nd", "+0.0005", "sokfirma.no")
        col3.metric("Margin over 3rd", "+0.0017", "SingularIT")

        st.dataframe(
            df_lb.style.apply(
                lambda x: ["background-color: #2E7D32; color: white" if x.name == 0 else "" for _ in x],
                axis=1
            ),
            use_container_width=True, hide_index=True
        )

        st.subheader("Public vs Private Test Gap")
        st.markdown("""
        Our public leaderboard score was **0.9226** (4th place during the competition).
        On the private test set, we moved to **1st place** with 0.7113.
        This confirms that our generalization-first strategy paid off — teams that optimized for the public test dropped in the final ranking.
        """)

    # === Tab 2: Submission Journey ===
    with tab2:
        st.header("Submission Journey")
        df_sub = pd.DataFrame(LEADERBOARD)
        df_sub["delta"] = df_sub["public_test"].diff().fillna(0)
        df_sub.index = range(1, len(df_sub) + 1)
        df_sub.index.name = "#"

        st.line_chart(df_sub.set_index("submission")["public_test"], height=350)

        st.dataframe(df_sub, use_container_width=True)

        st.subheader("Key Transitions")
        st.markdown("""
        **Single → Ensemble (+0.129)**: The single biggest improvement came from WBF ensemble.
        3 weak-ish models combined beat any single strong model by a massive margin.

        **Same arch → Mixed arch (+0.005)**: Switching from 3x YOLOv8x to 2x YOLOv8x + 1x YOLOv8l
        improved the score. Different architectures make different errors.

        **Mixed arch → Triple arch (+0.0016)**: Adding YOLO11x (a completely different architecture family)
        gave the final edge. Three-axis diversity: architecture × augmentation × capacity.
        """)

    # === Tab 3: Training Runs ===
    with tab3:
        st.header("Training Runs (150+ experiments)")

        if Path(EXPERIMENTS_CSV).exists():
            df_exp = pd.read_csv(EXPERIMENTS_CSV)
            training = df_exp[df_exp["type"] == "training"].copy()

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Runs", len(training))
            col2.metric("Architectures", training["arch"].nunique() if "arch" in training.columns else "N/A")
            col3.metric("Seeds Used", training["seed"].nunique() if "seed" in training.columns else "N/A")

            if "arch" in training.columns:
                st.subheader("Runs by Architecture")
                arch_counts = training["arch"].value_counts()
                st.bar_chart(arch_counts)

            if "mAP50" in training.columns:
                st.subheader("mAP@0.5 Distribution (validation)")
                training["mAP50"] = pd.to_numeric(training["mAP50"], errors="coerce")
                valid_maps = training.dropna(subset=["mAP50"])
                if len(valid_maps) > 0:
                    st.bar_chart(valid_maps.set_index("experiment")["mAP50"].sort_values(ascending=False).head(30))

            if "imgsz" in training.columns:
                st.subheader("Runs by Image Size")
                st.bar_chart(training["imgsz"].value_counts())

            st.subheader("All Training Runs")
            display_cols = [c for c in ["experiment", "arch", "imgsz", "seed", "epochs", "mAP50", "test_score", "notes"] if c in training.columns]
            st.dataframe(training[display_cols].sort_values("mAP50", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.warning("experiments.csv not found")

    # === Tab 4: Local Evaluation ===
    with tab4:
        st.header("Local Evaluation Results")

        if Path(EVAL_FILE).exists():
            with open(EVAL_FILE) as f:
                evals = json.load(f)

            df_eval = pd.DataFrame(evals)
            df_eval = df_eval.sort_values("competition_score", ascending=False)

            col1, col2 = st.columns(2)
            col1.metric("Best Local Score", f"{df_eval['competition_score'].max():.4f}")
            col2.metric("Configs Evaluated", len(df_eval))

            st.subheader("Detection vs Classification Breakdown")
            st.markdown("""
            The competition score is `0.7 × det_mAP + 0.3 × cls_mAP`.
            Detection (localization) is strong across all models (~97%).
            Classification is the bottleneck (~85-88% locally).
            """)

            df_eval_display = df_eval[["name", "competition_score", "detection_map50", "classification_map50", "num_predictions"]].copy()
            df_eval_display.columns = ["Submission", "Score", "Det mAP@0.5", "Cls mAP@0.5", "Predictions"]

            # Filter out broken runs
            df_eval_display = df_eval_display[df_eval_display["Score"] > 0.5]

            st.dataframe(df_eval_display, use_container_width=True, hide_index=True)

            if len(df_eval_display) > 1:
                st.subheader("Detection vs Classification Scatter")
                chart_data = df_eval_display[["Det mAP@0.5", "Cls mAP@0.5"]].copy()
                st.scatter_chart(chart_data, x="Det mAP@0.5", y="Cls mAP@0.5", height=400)
        else:
            st.warning("eval_results.json not found")

    # === Tab 5: Technique Analysis ===
    with tab5:
        st.header("Technique Impact Analysis")

        techniques = pd.DataFrame([
            {"Technique": "WBF Ensemble (3 models)", "Local Delta": "+0.060", "Test Delta": "+0.129", "Verdict": "Essential", "Generalized": True},
            {"Technique": "Multi-scale (640+800)", "Local Delta": "+0.008", "Test Delta": "+0.005", "Verdict": "Strong", "Generalized": True},
            {"Technique": "TTA Horizontal Flip", "Local Delta": "+0.003", "Test Delta": "+0.002", "Verdict": "Helpful", "Generalized": True},
            {"Technique": "Architecture Diversity", "Local Delta": "+0.002", "Test Delta": "+0.002", "Verdict": "Winning edge", "Generalized": True},
            {"Technique": "Full data (no val split)", "Local Delta": "N/A", "Test Delta": "+0.004", "Verdict": "Critical", "Generalized": True},
            {"Technique": "flipud=0.0", "Local Delta": "+0.004", "Test Delta": "+0.001", "Verdict": "Small gain", "Generalized": True},
            {"Technique": "absent_model_aware_avg", "Local Delta": "+0.001", "Test Delta": "+0.001", "Verdict": "Small gain", "Generalized": True},
            {"Technique": "cls=2.0 loss weight", "Local Delta": "+0.015", "Test Delta": "+0.000", "Verdict": "OVERFITS", "Generalized": False},
            {"Technique": "Two-stage classifier", "Local Delta": "+0.007", "Test Delta": "-0.011", "Verdict": "OVERFITS", "Generalized": False},
            {"Technique": "Gallery re-ranking", "Local Delta": "+0.000", "Test Delta": "-0.003", "Verdict": "Hurts", "Generalized": False},
            {"Technique": "Model soup", "Local Delta": "-0.015", "Test Delta": "N/A", "Verdict": "Kills diversity", "Generalized": False},
            {"Technique": "Neighbor voting", "Local Delta": "-0.004", "Test Delta": "N/A", "Verdict": "Too aggressive", "Generalized": False},
            {"Technique": "Soft-NMS", "Local Delta": "-0.001", "Test Delta": "N/A", "Verdict": "No gain", "Generalized": False},
            {"Technique": "960px 3rd scale", "Local Delta": "+0.000", "Test Delta": "-0.001", "Verdict": "Diminishing returns", "Generalized": False},
        ])

        gen = techniques[techniques["Generalized"] == True]
        no_gen = techniques[techniques["Generalized"] == False]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Techniques that Generalized")
            st.dataframe(gen[["Technique", "Local Delta", "Test Delta", "Verdict"]], use_container_width=True, hide_index=True)

        with col2:
            st.subheader("Techniques that Failed to Generalize")
            st.dataframe(no_gen[["Technique", "Local Delta", "Test Delta", "Verdict"]], use_container_width=True, hide_index=True)

        st.subheader("The Generalization Trap")
        st.markdown("""
        The most dangerous techniques are those with **large local improvements but zero test improvement**.
        `cls=2.0` showed +0.015 locally — a massive gain that would fool most practitioners into keeping it.
        But it showed exactly 0.000 improvement on test, meaning it memorized training class distributions
        rather than learning generalizable features.

        **The courage to reject locally-validated improvements was the single most important factor in winning.**
        """)

        st.subheader("Error Breakdown: Why Classification is Hard")
        errors = pd.DataFrame([
            {"Error Type": "Same-brand confusion (e.g. Nescafé 100g vs 200g)", "Share": "74%"},
            {"Error Type": "Small object misses (<32px at inference)", "Share": "15%"},
            {"Error Type": "Single-instance classes (can't learn from 1 example)", "Share": "6%"},
            {"Error Type": "Occlusion / partial visibility", "Share": "5%"},
        ])
        st.dataframe(errors, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
