from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx

RES = Path(__file__).resolve().parent.parent / "results"
CLASSES = ["Basal", "Her2", "LumA", "LumB", "Normal"]
MODELS = ["LogisticRegression", "RandomForest", "MLP", "HeteroGNN_v1", "HeteroGNN_v2_classweighted"]
CM_FILES = {"LogisticRegression": "cm_LogisticRegression.png", "RandomForest": "cm_RandomForest.png",
            "MLP": "cm_MLP.png", "HeteroGNN_v1": "cm_HeteroGNN.png",
            "HeteroGNN_v2_classweighted": "cm_HeteroGNN_v2.png"}
PC_FILES = {"LogisticRegression": "per_class_LogisticRegression.csv", "RandomForest": "per_class_RandomForest.csv",
            "MLP": "per_class_MLP.csv", "HeteroGNN_v1": "per_class_HeteroGNN.csv",
            "HeteroGNN_v2_classweighted": "per_class_HeteroGNN_v2.csv"}
METHODS = {"Random Forest (SHAP)": "RF_SHAP", "GNN v2 (gradient x input)": "GNN_gradxinput"}

st.set_page_config(page_title="CancerGraph", layout="wide")


@st.cache_data
def load_csv(name, index_col=None):
    p = RES / name
    return pd.read_csv(p, index_col=index_col) if p.exists() else None


st.title("CancerGraph")
st.caption("Heterogeneous graph neural network vs. standard ML for breast cancer PAM50 subtype "
           "classification (TCGA-BRCA). Student research project.")
st.warning("Research output only. These are model predictions for existing test samples, "
           "NOT a clinical diagnosis and not for medical decisions.")

comparison = load_csv("model_comparison.csv")
preds = load_csv("test_predictions_all.csv")
rank = load_csv("gene_rankings.csv")
markers = load_csv("known_marker_check.csv")
edges = load_csv("string_edges_top1000.csv")

tabs = st.tabs(["Candidate biomarkers", "Model comparison", "Confusion matrix", "Predictions per sample",
                "Top genes", "Gene network", "About and limitations"])

# ---------------- Tab 0: candidate biomarkers ----------------
KNOWN = {"ESR1": "luminal", "PGR": "luminal", "GATA3": "luminal", "FOXA1": "luminal",
         "ERBB2": "HER2", "KRT5": "basal", "KRT14": "basal"}
with tabs[0]:
    st.subheader("Candidate biomarker genes per subtype")
    st.warning("These are CANDIDATE biomarkers: genes that the models relied on. They are not validated "
               "biomarkers and do not show causation. Validation would need independent data and experiments.")
    if rank is None:
        st.error("results/gene_rankings.csv not found.")
    else:
        c1, c2 = st.columns(2)
        subtype = c1.selectbox("Subtype", CLASSES, key="bio_subtype")
        n = c2.slider("Top N genes per method", 5, 20, 10, key="bio_n")
        if subtype == "Normal":
            st.info("Only 3 Normal samples were in the test set, so these rankings are not interpretable.")

        sub = rank[rank["class"] == subtype]
        rf = sub[sub["method"] == "RF_SHAP"].set_index("gene")["rank"]
        gn = sub[sub["method"] == "GNN_gradxinput"].set_index("gene")["rank"]
        gdir = {}
        if "direction" in sub.columns:
            gdir = sub[sub["method"] == "GNN_gradxinput"].set_index("gene")["direction"].to_dict()
        rf_top = set(rf[rf <= n].index)
        gn_top = set(gn[gn <= n].index)

        rows = []
        for g in sorted(rf_top | gn_top):
            in_rf, in_gn = g in rf_top, g in gn_top
            rr = int(rf[g]) if g in rf.index else None
            gr = int(gn[g]) if g in gn.index else None
            rows.append({
                "gene": g,
                "evidence": "Both methods" if (in_rf and in_gn) else ("RF SHAP only" if in_rf else "GNN only"),
                "RF rank": str(rr) if rr is not None else ">20",
                "GNN rank": str(gr) if gr is not None else ">20",
                "GNN direction": str(gdir.get(g, "")),
                "known marker": f"Known {KNOWN[g]} marker" if g in KNOWN else "",
                "_best": min(x for x in (rr, gr) if x is not None),
            })
        df = pd.DataFrame(rows)
        df["_o"] = df["evidence"].map({"Both methods": 0, "RF SHAP only": 1, "GNN only": 2})
        df = df.sort_values(["_o", "_best"]).drop(columns=["_o", "_best"])

        m1, m2 = st.columns(2)
        m1.metric("Genes found by both methods", int((df["evidence"] == "Both methods").sum()))
        m2.metric("Genes listed", len(df))
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption("Evidence: 'Both methods' = Random Forest SHAP and GNN gradient both rank the gene in their top N. "
                   "'GNN only' is the weakest evidence: in this run the GNN rankings agreed poorly with the Random Forest "
                   "and with known markers. Importance values are absolute for the Random Forest, so its direction is unknown. "
                   "Several known markers (e.g. ESR1, FOXA1) are part of the PAM50 gene set that defines the labels, "
                   "so finding them is a sanity check, not a discovery.")

        if markers is not None:
            st.markdown("**Known-marker check** (rank among 1,000 genes; 1 = most important)")
            st.dataframe(markers, use_container_width=True, hide_index=True)
            st.caption("ERBB2 was not among the 1,000 most variable genes, so the models never saw it.")

# ---------------- Tab 1: model comparison ----------------
with tabs[1]:
    st.subheader("Model comparison (test set, 127 samples)")
    if comparison is None:
        st.error("results/model_comparison.csv not found.")
    else:
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        st.bar_chart(comparison.set_index("model")[["accuracy", "macro_f1", "balanced_accuracy"]])
        best = comparison.sort_values("macro_f1").iloc[-1]["model"]
        st.info(f"Highest macro-F1 on the test set: **{best}**. The last column is a bootstrap "
                "range: where ranges overlap, the difference between models may be just noise "
                "from the small test set. The GNN is not assumed to win.")

# ---------------- Tab 2: confusion matrix ----------------
with tabs[2]:
    st.subheader("Confusion matrix and per-class results (test set)")
    model = st.selectbox("Model", MODELS, key="cm_model")
    left, right = st.columns(2)
    img = RES / CM_FILES[model]
    if img.exists():
        left.image(str(img), caption="Rows = true subtype, columns = predicted subtype")
    else:
        left.error(f"{img.name} not found in results/")
    pc = load_csv(PC_FILES[model], index_col=0)
    if pc is not None:
        right.dataframe(pc, use_container_width=True)
    st.caption("The Normal class has only 3 test samples, so its precision and recall are unreliable.")

# ---------------- Tab 3: predictions per sample ----------------
with tabs[3]:
    st.subheader("Predicted subtype for each test sample")
    st.caption("The GNN is transductive: it predicts samples that are already in the graph. "
               "New samples cannot be uploaded.")
    if preds is None:
        st.error("results/test_predictions_all.csv not found.")
    else:
        c1, c2, c3 = st.columns(3)
        model = c1.selectbox("Model", MODELS, key="pred_model")
        subtype = c2.selectbox("True subtype", ["All"] + CLASSES)
        only_err = c3.checkbox("Show only mistakes")
        view = preds[["sample", "true", model]].rename(columns={model: "predicted"})
        view["correct"] = view["true"] == view["predicted"]
        if subtype != "All":
            view = view[view["true"] == subtype]
        if only_err:
            view = view[~view["correct"]]
        m1, m2 = st.columns(2)
        m1.metric("Samples shown", len(view))
        if len(view) > 0:
            m2.metric("Fraction correct (shown)", f"{view['correct'].mean():.2f}")
        st.dataframe(view, use_container_width=True, hide_index=True)

# ---------------- Tab 4: top genes ----------------
with tabs[4]:
    st.subheader("Top genes per subtype (candidate biomarkers)")
    if rank is None:
        st.error("results/gene_rankings.csv not found.")
    else:
        c1, c2, c3 = st.columns(3)
        method_label = c1.selectbox("Method", list(METHODS.keys()))
        subtype = c2.selectbox("Subtype", CLASSES, key="genes_subtype")
        n = c3.slider("Number of genes", 5, 20, 10)
        sel = rank[(rank["method"] == METHODS[method_label]) & (rank["class"] == subtype)]
        sel = sel.sort_values("rank").head(n)
        fig, ax = plt.subplots(figsize=(6, 0.35 * n + 1))
        ax.barh(sel["gene"][::-1], sel["importance"][::-1])
        ax.set_xlabel("importance")
        st.pyplot(fig)
        st.dataframe(sel, use_container_width=True, hide_index=True)
        if markers is not None:
            st.markdown("**Check against known subtype markers** (rank among 1,000 genes; 1 = most important)")
            st.dataframe(markers, use_container_width=True, hide_index=True)
        st.caption("Importance is not causality. Several markers (ESR1, ERBB2, KRT5, FOXA1...) are part of the "
                   "PAM50 gene set that defines the labels, so seeing them is a sanity check, not a discovery. "
                   "Other genes are only candidates needing independent validation. Rankings come from one seed "
                   "and one split.")

# ---------------- Tab 5: gene network ----------------
with tabs[5]:
    st.subheader("STRING gene network around the top genes")
    if rank is None or edges is None:
        st.error("results/gene_rankings.csv or results/string_edges_top1000.csv not found.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        method_label = c1.selectbox("Method", list(METHODS.keys()), key="net_method")
        subtype = c2.selectbox("Subtype", CLASSES, key="net_subtype")
        n = c3.slider("Top genes", 5, 30, 15, key="net_n")
        show_nb = c4.checkbox("Add STRING neighbours")
        top = rank[(rank["method"] == METHODS[method_label]) & (rank["class"] == subtype)]
        top = top.sort_values("rank").head(n)["gene"].tolist()
        topset = set(top)

        G = nx.Graph()
        G.add_nodes_from(top)
        inside = edges[edges["gene_a"].isin(topset) & edges["gene_b"].isin(topset)]
        G.add_edges_from(zip(inside["gene_a"], inside["gene_b"]))

        neighbours = set()
        if show_nb:
            one_end = edges[edges["gene_a"].isin(topset) ^ edges["gene_b"].isin(topset)]
            others = [b if a in topset else a for a, b in zip(one_end["gene_a"], one_end["gene_b"])]
            neighbours = set(sorted(set(others))[:40])        # cap for readability
            for a, b in zip(one_end["gene_a"], one_end["gene_b"]):
                if a in neighbours or b in neighbours:
                    G.add_edge(a, b)

        if G.number_of_edges() == 0:
            st.info("No STRING edges (score >= 700) among these genes. This is possible: only part of "
                    "the 1,000 selected genes have any STRING edge. Try more genes or add neighbours.")
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = nx.spring_layout(G, seed=42)
        colors = ["#d62728" if g in topset else "#bbbbbb" for g in G.nodes()]
        nx.draw_networkx(G, pos, ax=ax, node_color=colors, node_size=450, font_size=8, edge_color="#888888")
        ax.axis("off")
        st.pyplot(fig)
        st.caption("Red = top genes for this subtype. Grey = STRING neighbours (if shown). STRING edges are "
                   "functional associations (combined score >= 700), not necessarily physical interactions.")

# ---------------- Tab 6: about ----------------
with tabs[6]:
    st.subheader("About and limitations")
    st.markdown("""
**Data:** TCGA-BRCA from UCSC Xena (RNA-seq, 844 primary tumours, PAM50 labels from the Xena clinical matrix).
**Graph:** sample nodes and gene nodes; sample-gene "expresses" edges and gene-gene STRING edges (score >= 700).
**Models:** Logistic Regression, Random Forest and MLP baselines, and a heterogeneous GNN (v1, and v2 with class weights).

**Limitations**
- Single split and seed; the test set has 127 samples and only 3 Normal samples.
- PAM50 labels are computed from gene expression, which limits what any model can add.
- GNN v2 was designed after seeing v1's results; baselines were not tuned the same way.
- STRING edges are associations, not proof of physical interaction.
- Importance is not causality. Top genes are candidate biomarkers only.
- The GNN is transductive, so it cannot predict new uploaded samples.
""")
