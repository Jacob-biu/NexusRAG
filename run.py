import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
from datetime import datetime
import os

warnings.filterwarnings("ignore")

# Project modules
from src.config import NexusRAGConfig
from src.NexusRAG import NexusRAG
from src.utils import LLM_Model, setup_logging
from src.evaluate import Evaluator
from sentence_transformers import SentenceTransformer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spacy_model",
        type=str,
        default="en_core_web_trf",
        help="The spacy model to use",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="model/all-mpnet-base-v2",
        help="The path of embedding model to use",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="novel", help="The dataset to use"
    )
    # QA model arguments
    parser.add_argument(
        "--llm_model", type=str, default="", help="QA LLM model name; falls back to the utils default when empty"
    )
    parser.add_argument(
        "--llm_base_url",
        type=str,
        default="",
        help="QA LLM service URL; falls back to the utils default when empty",
    )
    # Evaluation model arguments
    parser.add_argument(
        "--eval_llm_model",
        type=str,
        default="",
        help="dedicated evaluation LLM model name; reuses the QA model when empty",
    )
    parser.add_argument(
        "--eval_llm_base_url",
        type=str,
        default="",
        help="dedicated evaluation LLM service URL; reuses the QA model URL when empty",
    )

    parser.add_argument(
        "--max_workers", type=int, default=16, help="The max number of workers to use"
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=3,
        help="The max number of iterations to use",
    )
    parser.add_argument(
        "--iteration_threshold",
        type=float,
        default=0.4,
        help="The threshold for iteration",
    )
    parser.add_argument(
        "--top_k_sentence", type=int, default=3, help="The top k sentence to use"
    )
    parser.add_argument(
        "--use_vectorized_retrieval",
        action="store_true",
        help="Use vectorized matrix-based retrieval instead of BFS iteration",
    )
    parser.add_argument(
        "--top_k_entity_cooccur",
        type=int,
        default=5,
        help="top_k_entity_cooccur for limiting coccur_entity",
    )
    parser.add_argument(
        "--precompute_threshold",
        type=float,
        default=0.5,
        help="precompute_threshold for limiting the threshold of sentence and entity similarity",
    )

    parser.add_argument(
        "--cooccur_alpha",
        type=str,
        default=None,
        help="pass a single alpha to skip the sweep; otherwise sweep 0 to 1 in steps of 0.05 (21 runs)",
    )

    # Resume toggle (enabled by default)
    parser.add_argument(
        "--no_resume",
        dest="enable_resume",
        action="store_false",
        default=True,
        help="disable resume and keep the original logic (resume is on by default)",
    )
    parser.add_argument(
        "--resume_save_every",
        type=int,
        default=20,
        help="save predictions.json every N entries in resume mode",
    )

    # Retrieval-only mode (skips QA and evaluation)
    parser.add_argument(
        "--retrieval_only",
        action="store_true",
        default=False,
        help="run the retrieval stage only and save retrieval.json, skipping QA and evaluation (no alpha_sweep CSV)",
    )

    return parser.parse_args()


def load_dataset(dataset_name):
    questions_path = f"./import/dataset/{dataset_name}/questions.json"
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    chunks_path = f"./import/dataset/{dataset_name}/chunks.json"
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    passages = [f"{idx}:{chunk}" for idx, chunk in enumerate(chunks)]
    return questions, passages


def load_embedding_model(embedding_model):
    embedding_model = SentenceTransformer(embedding_model, device="cuda")
    return embedding_model


def run_single_experiment(
    args,
    questions,
    passages,
    embedding_model,
    qa_llm,
    eval_llm,
    alpha,
    base_result_dir,
):
    """Run single experiment"""

    time = datetime.now()
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    target_prefix = f"alpha_{alpha}_"
    reuse_dir = None
    copy_source_retrieval = None

    # 1. Scan base_result_dir for subdirectories sharing the alpha prefix
    if os.path.exists(base_result_dir):
        all_subdirs = [
            os.path.join(base_result_dir, d)
            for d in os.listdir(base_result_dir)
            if os.path.isdir(os.path.join(base_result_dir, d))
            and d.startswith(target_prefix)
        ]
        # Keep the directories that have retrieval.json but no predictions.json, they are reusable
        candidate_reuse = []
        candidate_copy_src = []
        for d in all_subdirs:
            ret_path = os.path.join(d, "retrieval.json")
            pred_path = os.path.join(d, "predictions.json")
            if os.path.exists(ret_path):
                if not os.path.exists(pred_path):
                    candidate_reuse.append(d)
                else:
                    candidate_copy_src.append(ret_path)
        # Prefer the first reusable directory
        if candidate_reuse:
            reuse_dir = candidate_reuse[0]
        # Record the first copyable retrieval.json (from an older directory that already has predictions)
        if candidate_copy_src:
            copy_source_retrieval = candidate_copy_src[0]

    # 2. Determine the final result_dir and retrieved_path
    if reuse_dir is not None:
        # Case 1: a reusable directory was found, reuse it directly
        result_dir = reuse_dir
        retrieved_path = os.path.join(result_dir, "retrieval.json")
    else:
        # Case 2/3: create a new directory
        result_dir = f"{base_result_dir}/{target_prefix}{time_str}"
        os.makedirs(result_dir, exist_ok=True)
        retrieved_path = os.path.join(result_dir, "retrieval.json")
        # Copy the old retrieval.json when one is available
        if copy_source_retrieval is not None and os.path.exists(copy_source_retrieval):
            import shutil

            shutil.copy2(copy_source_retrieval, retrieved_path)

    setup_logging(f"{result_dir}/log.txt")

    config = NexusRAGConfig(
        dataset_name=args.dataset_name,
        embedding_model=embedding_model,
        spacy_model=args.spacy_model,
        max_workers=args.max_workers,
        llm_model=qa_llm,
        max_iterations=args.max_iterations,
        iteration_threshold=args.iteration_threshold,
        top_k_sentence=args.top_k_sentence,
        use_vectorized_retrieval=args.use_vectorized_retrieval,
        top_k_entity_cooccur=args.top_k_entity_cooccur,
        cooccur_alpha=alpha,
        precompute_threshold=args.precompute_threshold,
    )

    rag_model = NexusRAG(global_config=config)
    rag_model.index(passages)
    retrieval_path = f"{result_dir}/retrieval.json"
    # Pass retrieved_path so that an existing retrieval file is reused
    questions_with_pred = rag_model.qa(questions, retrieval_path, retrieved_path)
    FAIL_FLAG = "LLM request failed (retry 3/3)"
    roundx = 0
    while roundx <= 5:
        roundx += 1
        fail_index = []
        for idx, item in enumerate(questions_with_pred):
            if FAIL_FLAG in item.get("pred_answer", ""):
                fail_index.append(idx)
        if not fail_index:
            break
        print(f"QA stage: {len(fail_index)} failed samples detected, re-running batch {roundx}")
        questions_with_pred = rag_model.re_run_failed_qa(
            questions_with_pred, fail_index
        )

    pred_path = f"{result_dir}/predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(questions_with_pred, f, ensure_ascii=False, indent=4)

    # Use a dedicated evaluation model
    evaluator = Evaluator(llm_model=eval_llm, predictions_path=pred_path)
    evaluator.max_workers = args.max_workers
    em, contain_acc = evaluator.evaluate(max_workers=args.max_workers)

    return em, contain_acc, result_dir


# Resume-aware single-experiment function (independent of run_single_experiment)
def find_reuse_dir_for_resume(base_result_dir, target_prefix):
    """
    Find a reusable result_dir in resume mode:
    prefer a directory that contains retrieval.json so that the retrieval results are reused
    """
    if not os.path.exists(base_result_dir):
        return None
    all_subdirs = [
        os.path.join(base_result_dir, d)
        for d in os.listdir(base_result_dir)
        if os.path.isdir(os.path.join(base_result_dir, d))
        and d.startswith(target_prefix)
    ]
    if not all_subdirs:
        return None
    # Prefer a directory that contains retrieval.json
    for d in all_subdirs:
        if os.path.exists(os.path.join(d, "retrieval.json")):
            return d
    # Otherwise take the first one
    return all_subdirs[0]


def run_single_experiment_with_resume(
    args,
    questions,
    passages,
    embedding_model,
    qa_llm,
    eval_llm,
    alpha,
    base_result_dir,
    save_every=20,
):
    """
    Resume-aware single-experiment function, checking three stages in order:
    evaluation_results.json -> retrieval.json -> predictions.json
    1. evaluation_results.json exists -> skip this alpha
    2. retrieval.json missing -> run the full retrieval
    3. predictions.json:
       - missing -> run the full QA through qa_with_resume
       - exists with entries lacking pred_answer -> keep filling them through qa_with_resume
       - every entry has pred_answer -> check llm_accuracy and evaluate the missing ones through evaluate_with_resume
    """
    time = datetime.now()
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    target_prefix = f"alpha_{alpha}_"

    # 1. Locate or create result_dir
    result_dir = find_reuse_dir_for_resume(base_result_dir, target_prefix)
    if result_dir is None:
        result_dir = f"{base_result_dir}/{target_prefix}{time_str}"
    os.makedirs(result_dir, exist_ok=True)

    retrieval_path = os.path.join(result_dir, "retrieval.json")
    predictions_path = os.path.join(result_dir, "predictions.json")
    eval_result_path = os.path.join(result_dir, "evaluation_results.json")

    setup_logging(f"{result_dir}/log.txt")

    print(f"\n========== [Resume] alpha={alpha} ==========")
    print(f"[Resume] result_dir = {result_dir}")

    # 2. Stage 1: skip when evaluation_results.json already exists
    if os.path.exists(eval_result_path):
        with open(eval_result_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        llm_acc_str = eval_data.get("llm_accuracy", "0%")
        contain_str = eval_data.get("contain_accuracy", "0%")
        try:
            llm_acc = float(str(llm_acc_str).replace("%", ""))
        except (ValueError, TypeError):
            llm_acc = 0.0
        try:
            contain_acc = float(str(contain_str).replace("%", ""))
        except (ValueError, TypeError):
            contain_acc = 0.0
        print(
            f"[Resume] alpha={alpha} evaluation_results.json already exists, skipping "
            f"(EM={llm_acc:.2f}, Contain_Acc={contain_acc:.2f})"
        )
        return llm_acc, contain_acc, result_dir

    # 3. Build config and model (needed whether or not retrieval.json exists)
    config = NexusRAGConfig(
        dataset_name=args.dataset_name,
        embedding_model=embedding_model,
        spacy_model=args.spacy_model,
        max_workers=args.max_workers,
        llm_model=qa_llm,
        max_iterations=args.max_iterations,
        iteration_threshold=args.iteration_threshold,
        top_k_sentence=args.top_k_sentence,
        use_vectorized_retrieval=args.use_vectorized_retrieval,
        top_k_entity_cooccur=args.top_k_entity_cooccur,
        cooccur_alpha=alpha,
        precompute_threshold=args.precompute_threshold,
    )
    rag_model = NexusRAG(global_config=config)
    rag_model.index(passages)

    # 4. Stage 2: QA (resume-aware, reuses retrieval.json and predictions.json)
    questions_with_pred = rag_model.qa_with_resume(
        questions,
        retrieval_path,
        retrieval_path,
        predictions_path,
        save_every=save_every,
    )

    # 5. Stage 3: evaluation (resume-aware, targets entries missing llm_accuracy)
    evaluator = Evaluator(llm_model=eval_llm, predictions_path=predictions_path)
    em, contain_acc = evaluator.evaluate_with_resume(
        max_workers=args.max_workers, save_every=save_every
    )

    return em, contain_acc, result_dir


# Retrieval-only mode (skips QA and evaluation)
def run_retrieval_only(
    args,
    questions,
    passages,
    embedding_model,
    alpha,
    base_result_dir,
):
    """
    Retrieval-only mode: return as soon as retrieval.json exists.
    1. Reuse a directory under base_result_dir with the same alpha prefix that already holds retrieval.json
    2. Otherwise create a new directory, build the model, run retrieve() and save retrieval.json
    3. Skip QA and evaluation, do not write alpha_sweep_results.csv
    """
    time = datetime.now()
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    target_prefix = f"alpha_{alpha}_"

    # Reuse or create result_dir
    result_dir = find_reuse_dir_for_resume(base_result_dir, target_prefix)
    if result_dir is None:
        result_dir = f"{base_result_dir}/{target_prefix}{time_str}"
    os.makedirs(result_dir, exist_ok=True)

    retrieval_path = os.path.join(result_dir, "retrieval.json")
    setup_logging(f"{result_dir}/log.txt")

    print(f"\n========== [Retrieval-Only] alpha={alpha} ==========")
    print(f"[Retrieval-Only] result_dir = {result_dir}")

    # 1) Reuse retrieval.json when it already exists
    if os.path.exists(retrieval_path):
        with open(retrieval_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        print(f"[Retrieval-Only] Reusing the existing retrieval.json ({len(existing)} entries), skipping")
        return result_dir

    # 2) Run the full retrieval pass
    config = NexusRAGConfig(
        dataset_name=args.dataset_name,
        embedding_model=embedding_model,
        spacy_model=args.spacy_model,
        max_workers=args.max_workers,
        llm_model=None,
        max_iterations=args.max_iterations,
        iteration_threshold=args.iteration_threshold,
        top_k_sentence=args.top_k_sentence,
        use_vectorized_retrieval=args.use_vectorized_retrieval,
        top_k_entity_cooccur=args.top_k_entity_cooccur,
        cooccur_alpha=alpha,
        precompute_threshold=args.precompute_threshold,
    )
    rag_model = NexusRAG(global_config=config)
    rag_model.index(passages)

    retrieval_results = rag_model.retrieve(questions)
    with open(retrieval_path, "w", encoding="utf-8") as f:
        json.dump(retrieval_results, f, ensure_ascii=False, indent=2)
    print(
        f"[Retrieval-Only] Retrieval finished and saved ({len(retrieval_results)} entries) -> {retrieval_path}"
    )
    return result_dir


def update_visualization(df, output_dir):
    """Update visualization with English labels and data annotation"""
    df_valid = df.dropna(subset=["EM"])
    if len(df_valid) > 0:
        df_valid = df_valid.sort_values("cooccur_alpha").reset_index(drop=True)

        # Line plot: effect of alpha on EM, with value annotations
        plt.figure(figsize=(14, 6))
        x = df_valid["cooccur_alpha"]
        y = df_valid["EM"]
        plt.plot(x, y, marker="o", linewidth=2, color="#2E86AB", markersize=6)

        # Annotate each point with its value
        for xi, yi in zip(x, y):
            plt.annotate(
                f"{yi:.2f}",
                (xi, yi),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )

        plt.title("Alpha Sensitivity - EM Score", fontsize=14)
        plt.xlabel("cooccur_alpha", fontsize=12)
        plt.ylabel("EM Score", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/alpha_sweep_lineplot.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # Bar plot: effect of alpha on EM, with values labelled on top of the bars
        plt.figure(figsize=(16, 6))
        ax = sns.barplot(x="cooccur_alpha", y="EM", data=df_valid, palette="Blues_d")
        plt.xticks(rotation=45)

        # Add value labels on top of each bar
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(
                f"{height:.2f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=7,
            )

        plt.title("Alpha Sensitivity Bar Plot - EM Score", fontsize=14)
        plt.xlabel("cooccur_alpha", fontsize=12)
        plt.ylabel("EM Score", fontsize=12)
        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/alpha_sweep_barplot.png", dpi=300, bbox_inches="tight"
        )
        plt.close()


def main():
    args = parse_arguments()

    # Main output directory
    base_result_dir = f"results/alpha_sweep_k{args.top_k_entity_cooccur}_{args.llm_model}/threshold_{args.precompute_threshold}/{args.dataset_name}"
    os.makedirs(base_result_dir, exist_ok=True)

    # Aggregated CSV path
    result_csv_path = f"{base_result_dir}/alpha_sweep_results.csv"

    # Resume mode notice
    if args.enable_resume:
        print(f"🔁 Resume mode enabled (saving every {args.resume_save_every} entries)")
    else:
        print("⏹  Resume mode disabled (--no_resume), using the original logic")
    if args.retrieval_only:
        print("📥 Retrieval-only mode (--retrieval_only): skipping QA and evaluation, no alpha_sweep CSV")
    try:
        # 1. Global loading
        print("Loading global data and model...")
        embedding_model = load_embedding_model(args.embedding_model)
        questions, passages = load_dataset(args.dataset_name)

        # Initialize the QA LLM
        if args.llm_model:
            qa_llm = LLM_Model(llm_model=args.llm_model, base_url=args.llm_base_url)
        else:
            # Fall back to the defaults defined in utils when empty
            qa_llm = LLM_Model(llm_model="", base_url=None)

        # Initialize the evaluation LLM: use dedicated arguments when given, otherwise reuse the QA LLM
        if args.eval_llm_model:
            eval_llm = LLM_Model(
                llm_model=args.eval_llm_model, base_url=args.eval_llm_base_url
            )
        else:
            eval_llm = qa_llm

        # 2. Load existing results (core resume logic)
        existing_results = []
        done_params = set()

        if os.path.exists(result_csv_path):
            print(f"📂 Found existing result file: {result_csv_path}")
            print(f"⏳ Loading completed experiments...")
            df_existing = pd.read_csv(result_csv_path, encoding="utf-8")

            # Collect the completed parameter sets from the loaded CSV
            done_params = set()
            existing_results = []
            for _, row in df_existing.iterrows():
                param_key = (float(row["cooccur_alpha"]),)
                done_params.add(param_key)
                existing_results.append(row.to_dict())

            print(
                f"✅ Loaded {len(done_params)} completed experiments from the CSV."
            )

        # 3. Sweep alpha only
        if args.cooccur_alpha is not None:
            # A single alpha was given: skip the sweep and run only this one
            _raw = str(args.cooccur_alpha).strip()
            if "," in _raw:
                full_alpha_list = [round(float(x), 2) for x in _raw.split(",") if x.strip() != ""]
            else:
                full_alpha_list = [round(float(_raw), 2)]
            print(f"🔧 Single alpha={args.cooccur_alpha} given, skipping the sweep")
        else:
            # Default sweep: 0 to 1 in steps of 0.05 (21 values)
            full_alpha_list = [round(x * 0.05, 2) for x in range(0, 21)]
            print("Alpha list:", full_alpha_list)

        # Skip the runs that are already done
        all_params = []
        for alpha in full_alpha_list:
            param_key = (alpha,)
            if param_key not in done_params:
                all_params.append(alpha)

        if len(all_params) == 0:
            print("All experiments are completed!")
            # Update the plots
            df = pd.DataFrame(existing_results)
            update_visualization(df, base_result_dir)
            return

        print(f"Remaining {len(all_params)} experiments to run, starting...")

        # 4. Run
        pbar = tqdm(total=len(all_params), desc="Alpha Sweep Progress")
        for alpha in all_params:
            try:
                print(f"\nRunning: cooccur_alpha={alpha}")
                if args.retrieval_only:
                    # Retrieval-only mode: skip QA/evaluation and do not write the CSV
                    result_dir = run_retrieval_only(
                        args,
                        questions,
                        passages,
                        embedding_model,
                        alpha,
                        base_result_dir,
                    )
                    print(f"[Retrieval-Only] alpha={alpha} done -> {result_dir}")
                else:
                    if args.enable_resume:
                        # Resume mode: use the resume-aware function
                        em, contain_acc, result_dir = run_single_experiment_with_resume(
                            args,
                            questions,
                            passages,
                            embedding_model,
                            qa_llm,
                            eval_llm,
                            alpha,
                            base_result_dir,
                            save_every=args.resume_save_every,
                        )
                    else:
                        # Resume disabled: keep the original logic
                        em, contain_acc, result_dir = run_single_experiment(
                            args,
                            questions,
                            passages,
                            embedding_model,
                            qa_llm,
                            eval_llm,
                            alpha,
                            base_result_dir,
                        )
                    new_result = {
                        "cooccur_alpha": alpha,
                        "top_k_entity_cooccur": args.top_k_entity_cooccur,
                        "EM": em,
                        "Contain_Acc": contain_acc,
                        "result_dir": result_dir,
                    }
                    existing_results.append(new_result)
                    df = pd.DataFrame(existing_results)
                    df.to_csv(result_csv_path, index=False, encoding="utf-8")
                    update_visualization(df, base_result_dir)
                    print(f"Result saved, visualization updated")
            except Exception as e:
                print(f"Experiment failed: {e}")
                import traceback

                traceback.print_exc()
                if not args.retrieval_only:
                    new_result = {
                        "cooccur_alpha": alpha,
                        "top_k_entity_cooccur": args.top_k_entity_cooccur,
                        "EM": np.nan,
                        "Contain_Acc": np.nan,
                        "result_dir": None,
                    }
                    existing_results.append(new_result)
                    df = pd.DataFrame(existing_results)
                    df.to_csv(result_csv_path, index=False, encoding="utf-8")
            pbar.update(1)
        pbar.close()

        # 5. Final
        print("\nAll experiments completed!")
        df = pd.DataFrame(existing_results)
        df_valid = df.dropna(subset=["EM"])
        if len(df_valid) > 0:
            best_row = df_valid.loc[df_valid["EM"].idxmax()]

            print("\n======================")
            print("Best Parameter:")
            print(f"cooccur_alpha = {best_row['cooccur_alpha']}")
            print(f"Best EM Score: {best_row['EM']:.2f}")
            print(f"Best Contain Accuracy: {best_row['Contain_Acc']:.2f}")
            print("======================")

    except Exception as e:
        import traceback

        error_msg = traceback.format_exc()
        print(f"Program aborted: {str(e)}")
        print(error_msg)

        if existing_results:
            df = pd.DataFrame(existing_results)
            print(df.to_string(index=False))


if __name__ == "__main__":
    main()
