"""
evaluate.py  —  RAG Evaluation Suite for Lumina
================================================
Runs offline evaluation of the Lumina RAG pipeline on a test PDF.

Usage:
    python evaluate.py --pdf GRU.pdf

Requirements:
    pip install langchain-community langchain-mistralai chromadb
        pypdf rouge-score sentence-transformers python-dotenv
"""

import os, time, json, argparse, tempfile
from dataclasses import dataclass, field, asdict
from typing import List

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 ▸ TEST DATASET
#  Ground-truth QA pairs derived from GRU.pdf
# ─────────────────────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "question": "What is a Gated Recurrent Unit (GRU)?",
        "ground_truth": (
            "GRU is a simplified variant of LSTM proposed by Cho et al. in 2014. "
            "It reduces the number of gates from four to three by introducing a reset gate, "
            "an update gate, and a new memory cell, making it less complex than LSTM "
            "while still being effective for sequence learning."
        ),
    },
    {
        "question": "What are the three gates in a GRU cell?",
        "ground_truth": (
            "The three components in a GRU cell are: the reset gate, the update gate, "
            "and the new memory cell."
        ),
    },
    {
        "question": "How does the update gate in GRU work?",
        "ground_truth": (
            "The update gate controls how much of the previous hidden state is carried "
            "forward versus how much new information is incorporated. When the update gate "
            "value is close to one, the network forgets the previous state; when close to "
            "zero, it retains the previous state."
        ),
    },
    {
        "question": "What problem does LSTM solve that vanilla RNN cannot?",
        "ground_truth": (
            "LSTM solves the vanishing and exploding gradient problem in vanilla RNNs, "
            "which makes it difficult to learn long-term dependencies. LSTM uses gating "
            "mechanisms to selectively remember or forget information over long sequences."
        ),
    },
    {
        "question": "What is the minimal gated unit in GRU?",
        "ground_truth": (
            "The minimal gated unit is a simplified GRU variant proposed by Heck and Salem "
            "in 2017. It merges the reset gate and update gate into a single forget gate, "
            "further reducing the number of parameters while retaining core functionality."
        ),
    },
    {
        "question": "What is backpropagation through time (BPTT)?",
        "ground_truth": (
            "BPTT is a training algorithm for RNNs that extends standard backpropagation "
            "by applying the chain rule through time steps. The loss is computed as a "
            "sum of losses across previous time steps and gradients are propagated "
            "backwards through the unrolled network."
        ),
    },
    {
        "question": "What is the role of the reset gate in GRU?",
        "ground_truth": (
            "The reset gate controls how much of the previous hidden state is used when "
            "computing the new memory cell. It determines how much past information to "
            "forget or reset when processing new input."
        ),
    },
    {
        "question": "How is GRU different from LSTM?",
        "ground_truth": (
            "GRU is simpler than LSTM: it has three components instead of four gates, "
            "it does not have a separate cell state (only a hidden state), and it has "
            "fewer parameters. GRU merges the forget and input gates into an update gate "
            "and combines the cell state and hidden state."
        ),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 ▸ PIPELINE SETUP
#  Replicates the same pipeline as app.py
# ─────────────────────────────────────────────────────────────────────────────

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
 """You are Lumina. Answer the question using ONLY the provided context.
Be concise and direct. Do not add information not present in the context.
Start your answer immediately without preamble."""),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
])


def build_pipeline(pdf_path: str):
    """Load PDF, chunk, embed and return (retriever, llm)."""
    docs   = PyPDFLoader(pdf_path).load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=200
    ).split_documents(docs)

    persist_dir = os.path.join(tempfile.gettempdir(), "lumina_eval_chroma")
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=MistralAIEmbeddings(),
        persist_directory=persist_dir,
    )
    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.7},
    )
    llm = ChatMistralAI(model="mistral-small-2506")
    print(f"[pipeline] Loaded {len(docs)} pages → {len(chunks)} chunks")
    return retriever, llm


def run_query(question: str, retriever, llm) -> dict:
    """Run one RAG query; return answer + retrieved docs."""
    retrieved = retriever.invoke(question)
    context   = "\n\n".join([d.page_content for d in retrieved])
    t0        = time.time()
    response  = llm.invoke(PROMPT.invoke({"context": context, "question": question}))
    latency   = round(time.time() - t0, 3)
    return {
        "answer":    response.content,
        "context":   context,
        "retrieved": [d.page_content for d in retrieved],
        "latency":   latency,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 ▸ EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────

# Semantic similarity model (local, no API cost)
_sem_model = SentenceTransformer("all-MiniLM-L6-v2")
_rouge     = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)


def semantic_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity between two texts using a small sentence-transformer."""
    embs = _sem_model.encode([text_a, text_b], convert_to_tensor=True)
    return round(float(util.cos_sim(embs[0], embs[1])), 4)


def rouge_scores(prediction: str, reference: str) -> dict:
    """ROUGE-1 and ROUGE-L F1 scores."""
    scores = _rouge.score(reference, prediction)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def faithfulness_score(answer: str, context: str) -> float:
    """
    Proxy faithfulness: semantic similarity between the answer and
    the retrieved context. High score → answer is grounded in context.
    """
    return semantic_similarity(answer, context)


def context_relevance_score(question: str, retrieved_chunks: List[str]) -> float:
    """
    Average semantic similarity between the question and each retrieved chunk.
    Measures whether the retriever fetched relevant passages.
    """
    if not retrieved_chunks:
        return 0.0
    scores = [semantic_similarity(question, chunk) for chunk in retrieved_chunks]
    return round(sum(scores) / len(scores), 4)


def answer_relevance_score(question: str, answer: str) -> float:
    """Semantic similarity between the question and the answer."""
    return semantic_similarity(question, answer)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4 ▸ RUNNER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    question:           str
    ground_truth:       str
    predicted_answer:   str
    latency_s:          float
    semantic_sim:       float   # answer vs ground truth
    rouge1:             float
    rougeL:             float
    faithfulness:       float   # answer vs context
    context_relevance:  float   # question vs retrieved chunks
    answer_relevance:   float   # question vs answer
    retrieved_count:    int


def evaluate(pdf_path: str, output_json: str = "eval_results.json"):
    print("\n══════════════════════════════════════════════")
    print("  Lumina RAG — Evaluation Suite")
    print("══════════════════════════════════════════════\n")

    retriever, llm = build_pipeline(pdf_path)
    results: List[EvalResult] = []

    for i, tc in enumerate(TEST_CASES, 1):
        q  = tc["question"]
        gt = tc["ground_truth"]
        print(f"[{i}/{len(TEST_CASES)}] {q[:70]}…")

        out   = run_query(q, retriever, llm)
        pred  = out["answer"]
        ctx   = out["context"]
        chunks = out["retrieved"]

        sem  = semantic_similarity(pred, gt)
        rg   = rouge_scores(pred, gt)
        faith = faithfulness_score(pred, ctx)
        cr   = context_relevance_score(q, chunks)
        ar   = answer_relevance_score(q, pred)

        r = EvalResult(
            question=q, ground_truth=gt, predicted_answer=pred,
            latency_s=out["latency"],
            semantic_sim=sem,
            rouge1=rg["rouge1"], rougeL=rg["rougeL"],
            faithfulness=faith,
            context_relevance=cr,
            answer_relevance=ar,
            retrieved_count=len(chunks),
        )
        results.append(r)
        print(f"     semantic_sim={sem}  faithfulness={faith}  latency={out['latency']}s\n")

    # ── Aggregate ──────────────────────────────────────────────────────────
    def avg(key): return round(sum(getattr(r, key) for r in results) / len(results), 4)

    summary = {
        "total_questions":        len(results),
        "avg_semantic_similarity": avg("semantic_sim"),
        "avg_rouge1":             avg("rouge1"),
        "avg_rougeL":             avg("rougeL"),
        "avg_faithfulness":       avg("faithfulness"),
        "avg_context_relevance":  avg("context_relevance"),
        "avg_answer_relevance":   avg("answer_relevance"),
        "avg_latency_s":          avg("latency_s"),
    }

    print("\n──────────────────────────────────────────────")
    print("  AGGREGATE RESULTS")
    print("──────────────────────────────────────────────")
    for k, v in summary.items():
        print(f"  {k:<32} {v}")

    # ── Save ───────────────────────────────────────────────────────────────
    payload = {"summary": summary, "per_question": [asdict(r) for r in results]}
    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Full results saved → {output_json}")
    print("══════════════════════════════════════════════\n")
    return payload


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Lumina RAG pipeline")
    parser.add_argument("--pdf",    required=True, help="Path to test PDF")
    parser.add_argument("--output", default="eval_results.json", help="Output JSON path")
    args = parser.parse_args()
    evaluate(args.pdf, args.output)
