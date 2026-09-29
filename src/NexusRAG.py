from src.embedding_store import EmbeddingStore
from src.utils import min_max_normalize
import os
import json
from collections import defaultdict
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from src.ner import SpacyNER
import igraph as ig
import re
import logging
import torch
import pickle
from scipy.sparse import csr_matrix, save_npz, load_npz
logger = logging.getLogger(__name__)
class NexusRAG:
    def __init__(self, global_config):
        self.config = global_config
        logger.info(f"Initializing NexusRAG with config: {self.config}")
        retrieval_method = (
            "Vectorized Matrix-based"
            if self.config.use_vectorized_retrieval
            else "BFS Iteration"
        )
        logger.info(f"Using retrieval method: {retrieval_method}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.config.use_vectorized_retrieval:
            logger.info(f"Using device: {self.device} for vectorized retrieval")
        # Defaults for the improved pipeline, overridable through global_config
        self.cooccur_alpha = getattr(
            self.config, "cooccur_alpha", 0.1
        )  # Fusion coefficient between co-occurrence and semantic similarity
        self.top_k_entity_cooccur = getattr(
            self.config, "top_k_entity_cooccur", 5
        )  # Top-k sparsification for entity co-occurrence
        self.dataset_name = global_config.dataset_name
        self.load_embedding_store()
        self.llm_model = self.config.llm_model
        self.spacy_ner = SpacyNER(self.config.spacy_model)
        self.graph = ig.Graph(directed=False)
        # Precomputed-result cache paths
        self.cache_dir = os.path.join(
            self.config.working_dir,
            "v1",
            f"threshold_{self.config.precompute_threshold}",
            f"k{self.top_k_entity_cooccur}",
            self.dataset_name,
        )
        # Cache file paths
        self.cache_entity_cooccur = os.path.join(
            self.config.working_dir,
            "v1",
            self.dataset_name,
            "entity_cooccur_matrix.npz",
        )
        self.cache_entity_sim = os.path.join(
            self.config.working_dir, "v1", self.dataset_name, "entity_sim_matrix.npz"
        )
        self.cache_entity_neigh = os.path.join(
            self.cache_dir, "entity_cooccur_neighbors.pkl"
        )
        # ANN switch and hyperparameters (approximate nearest neighbor search replaces the O(N^2) full similarity matrix)
        self.use_ann_sim = getattr(self.config, "use_ann_sim", True)
        # Number of neighbors retrieved by ANN; the truncation K in the paper equals top_k_entity_cooccur
        self.ann_topk = getattr(self.config, "ann_topk", 10)
        # faiss index type: 'hnsw' (approximate and fast) or 'flat' (exact inner product)
        self.ann_index_type = getattr(self.config, "ann_index_type", "hnsw")
        self.cache_entity_sim_ann = os.path.join(
            self.config.working_dir,
            "v1",
            self.dataset_name,
            "entity_sim_matrix_ann.npz",
        )
        os.makedirs(self.cache_dir, exist_ok=True)
    def load_embedding_store(self):
        self.passage_embedding_store = EmbeddingStore(
            self.config.embedding_model,
            db_filename=os.path.join(
                self.config.working_dir,
                "v1",
                self.dataset_name,
                "passage_embedding.parquet",
            ),
            batch_size=self.config.batch_size,
            namespace="passage",
        )
        self.entity_embedding_store = EmbeddingStore(
            self.config.embedding_model,
            db_filename=os.path.join(
                self.config.working_dir,
                "v1",
                self.dataset_name,
                "entity_embedding.parquet",
            ),
            batch_size=self.config.batch_size,
            namespace="entity",
        )
        self.sentence_embedding_store = EmbeddingStore(
            self.config.embedding_model,
            db_filename=os.path.join(
                self.config.working_dir,
                "v1",
                self.dataset_name,
                "sentence_embedding.parquet",
            ),
            batch_size=self.config.batch_size,
            namespace="sentence",
        )
    def load_existing_data(self, passage_hash_ids):
        self.ner_results_path = os.path.join(
            self.config.working_dir, "v1", self.dataset_name, "ner_results.json"
        )
        if os.path.exists(self.ner_results_path):
            existing_ner_reuslts = json.load(open(self.ner_results_path))
            existing_passage_hash_id_to_entities = existing_ner_reuslts[
                "passage_hash_id_to_entities"
            ]
            existing_sentence_to_entities = existing_ner_reuslts["sentence_to_entities"]
            existing_passage_hash_ids = set(existing_passage_hash_id_to_entities.keys())
            new_passage_hash_ids = set(passage_hash_ids) - existing_passage_hash_ids
            return (
                existing_passage_hash_id_to_entities,
                existing_sentence_to_entities,
                new_passage_hash_ids,
            )
        else:
            return {}, {}, passage_hash_ids
    def qa(self, questions, retrieval_path, retrieved_path):
        if os.path.exists(retrieved_path):
            print(f"Reusing the existing retrieval file: {retrieved_path}, skipping the retrieval stage")
            with open(retrieved_path, "r", encoding="utf-8") as f:
                retrieval_results = json.load(f)
        else:
            # No cache available: run the retrieval and save the results
            retrieval_results = self.retrieve(questions)
            with open(retrieval_path, "w", encoding="utf-8") as f:
                json.dump(retrieval_results, f, ensure_ascii=False, indent=2)
        system_prompt = f"""As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations."""
        all_messages = []
        for retrieval_result in retrieval_results:
            question = retrieval_result["question"]
            sorted_passage = retrieval_result["sorted_passage"]
            prompt_user = """"""
            for passage in sorted_passage:
                prompt_user += f"{passage}\n"
            prompt_user += f"Question: {question}\n Thought: "
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_user},
            ]
            all_messages.append(messages)
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            all_qa_results = list(
                tqdm(
                    executor.map(self.llm_model.infer, all_messages),
                    total=len(all_messages),
                    desc="QA Reading (Parallel)",
                )
            )
        for qa_result, question_info in zip(all_qa_results, retrieval_results):
            try:
                pred_ans = qa_result.split("Answer:")[1].strip()
            except:
                pred_ans = qa_result
            question_info["pred_answer"] = pred_ans
        return retrieval_results
    # Dedicated helper that re-runs failed QA samples in batch
    def re_run_failed_qa(self, data_list, fail_index_list):
        system_prompt = """As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations."""
        re_messages = []
        for idx in fail_index_list:
            item = data_list[idx]
            q = item["question"]
            passages = item["sorted_passage"]
            user_text = ""
            for p in passages:
                user_text += f"{p}\n"
            user_text += f"Question: {q}\n Thought: "
            re_messages.append(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ]
            )
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            re_results = list(
                tqdm(
                    executor.map(self.llm_model.infer, re_messages),
                    total=len(re_messages),
                    desc="Re-running failed QA samples",
                )
            )
        # Overwrite the failed answers
        for pos, data_idx in enumerate(fail_index_list):
            raw_res = re_results[pos]
            try:
                new_pred = raw_res.split("Answer:")[1].strip()
            except:
                new_pred = raw_res
            data_list[data_idx]["pred_answer"] = new_pred
        return data_list
    # Resume-aware QA function (independent of qa(), the original logic is untouched)
    def qa_with_resume(
        self,
        questions,
        retrieval_path,
        retrieved_path,
        predictions_path,
        save_every=100,
    ):
        """
        Resume-aware QA:
        1. Reuse the retrieval results at retrieved_path (run the full retrieval when absent)
        2. Load an existing predictions.json and fill in the entries missing pred_answer
        3. Persist predictions.json every save_every entries
        4. Retry the failures at the end and save the final file
        """
        from concurrent.futures import as_completed
        # ---- 1) Retrieval stage: reuse retrieval.json when present, otherwise run it in full ----
        if os.path.exists(retrieved_path):
            print(f"[Resume-QA] Reusing the existing retrieval file: {retrieved_path}")
            with open(retrieved_path, "r", encoding="utf-8") as f:
                retrieval_results = json.load(f)
        else:
            print(f"[Resume-QA] {retrieved_path} not found, running the full retrieval")
            retrieval_results = self.retrieve(questions)
            with open(retrieval_path, "w", encoding="utf-8") as f:
                json.dump(retrieval_results, f, ensure_ascii=False, indent=2)
        # ---- 2) Load an existing predictions.json ----
        if os.path.exists(predictions_path):
            with open(predictions_path, "r", encoding="utf-8") as f:
                existing_predictions = json.load(f)
            print(
                f"[Resume-QA] Loaded existing predictions: {len(existing_predictions)} entries"
            )
        else:
            # Initial skeleton: templated from retrieval_results, every entry lacks pred_answer
            existing_predictions = [dict(item) for item in retrieval_results]
            for item in existing_predictions:
                item.pop("pred_answer", None)
            print(
                f"[Resume-QA] Initialized the predictions skeleton, {len(existing_predictions)} entries pending QA"
            )
        # ---- 3) Locate the entries missing pred_answer ----
        pending_indices = [
            idx
            for idx, item in enumerate(existing_predictions)
            if "pred_answer" not in item
        ]
        if not pending_indices:
            print("[Resume-QA] Every entry already has pred_answer, skipping the QA stage")
            return existing_predictions
        print(
            f"[Resume-QA] Entries to run QA for this round: {len(pending_indices)} / {len(existing_predictions)}"
        )
        # ---- 4) Build the messages, run inference concurrently and save incrementally ----
        system_prompt = (
            'As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. '
            'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
            'Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.'
        )
        idx_msg_pairs = []
        for idx in pending_indices:
            item = existing_predictions[idx]
            question = item["question"]
            sorted_passage = item["sorted_passage"]
            prompt_user = ""
            for passage in sorted_passage:
                prompt_user += f"{passage}\n"
            prompt_user += f"Question: {question}\n Thought: "
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_user},
            ]
            idx_msg_pairs.append((idx, messages))
        completed = 0
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_pair = {
                executor.submit(self.llm_model.infer, msg): (idx, msg)
                for idx, msg in idx_msg_pairs
            }
            for future in tqdm(
                as_completed(future_to_pair),
                total=len(idx_msg_pairs),
                desc="QA Reading (Resume)",
            ):
                idx, _ = future_to_pair[future]
                try:
                    qa_result = future.result()
                except Exception as e:
                    qa_result = f"LLM request failed (retry 3/3): {str(e)}"
                try:
                    pred_ans = qa_result.split("Answer:")[1].strip()
                except Exception:
                    pred_ans = qa_result
                existing_predictions[idx]["pred_answer"] = pred_ans
                completed += 1
                if completed % save_every == 0:
                    with open(predictions_path, "w", encoding="utf-8") as f:
                        json.dump(
                            existing_predictions, f, ensure_ascii=False, indent=4
                        )
                    print(
                        f"[Resume-QA] Incremental save: {completed}/{len(idx_msg_pairs)}"
                    )
        # Final save after this round completes
        with open(predictions_path, "w", encoding="utf-8") as f:
            json.dump(existing_predictions, f, ensure_ascii=False, indent=4)
        print(f"[Resume-QA] QA stage finished, {completed} entries run and saved")
        # ---- 5) Retry the failures (equivalent to the original logic) ----
        FAIL_FLAG = "LLM request failed (retry 3/3)"
        roundx = 0
        while roundx <= 5:
            roundx += 1
            fail_index = [
                idx
                for idx, item in enumerate(existing_predictions)
                if FAIL_FLAG in item.get("pred_answer", "")
            ]
            if not fail_index:
                break
            print(
                f"[Resume-QA] QA stage: {len(fail_index)} failed samples detected, re-running batch {roundx}"
            )
            existing_predictions = self.re_run_failed_qa(
                existing_predictions, fail_index
            )
            with open(predictions_path, "w", encoding="utf-8") as f:
                json.dump(existing_predictions, f, ensure_ascii=False, indent=4)
        return existing_predictions
    def retrieve(self, questions):
        self.entity_hash_ids = list(self.entity_embedding_store.hash_id_to_text.keys())
        self.entity_embeddings = np.array(self.entity_embedding_store.embeddings)
        self.passage_hash_ids = list(
            self.passage_embedding_store.hash_id_to_text.keys()
        )
        self.passage_embeddings = np.array(self.passage_embedding_store.embeddings)
        self.sentence_hash_ids = list(
            self.sentence_embedding_store.hash_id_to_text.keys()
        )
        self.sentence_embeddings = np.array(self.sentence_embedding_store.embeddings)
        self.node_name_to_vertex_idx = {
            v["name"]: v.index for v in self.graph.vs if "name" in v.attributes()
        }
        self.vertex_idx_to_node_name = {
            v.index: v["name"] for v in self.graph.vs if "name" in v.attributes()
        }
        # Precompute sparse matrices for vectorized retrieval if needed
        if self.config.use_vectorized_retrieval:
            logger.info(
                "Precomputing sparse adjacency matrices for vectorized retrieval..."
            )
            self._precompute_sparse_matrices()
            e2s_shape = self.entity_to_sentence_sparse.shape
            s2e_shape = self.sentence_to_entity_sparse.shape
            e2s_nnz = self.entity_to_sentence_sparse._nnz()
            s2e_nnz = self.sentence_to_entity_sparse._nnz()
            # Log the sparsity of the improved matrices
            w_shape = self.entity_weak_cooccur_W.shape
            w_nnz = self.entity_weak_cooccur_W._nnz()
            logger.info(
                f"Matrices built: Entity-Sentence {e2s_shape}, Sentence-Entity {s2e_shape}"
            )
            logger.info(
                f"Improved Matrices: Entity-Cooccur {w_shape}"
            )
            logger.info(
                f"E2S Sparsity: {(1 - e2s_nnz / (e2s_shape[0] * e2s_shape[1])) * 100:.2f}% (nnz={e2s_nnz})"
            )
            logger.info(
                f"S2E Sparsity: {(1 - s2e_nnz / (s2e_shape[0] * s2e_shape[1])) * 100:.2f}% (nnz={s2e_nnz})"
            )
            logger.info(
            )
            logger.info(
                f"W Sparsity: {(1 - w_nnz / (w_shape[0] * w_shape[1])) * 100:.2f}% (nnz={w_nnz})"
            )
            logger.info(f"Device: {self.device}")
        else:
            # The BFS mode also needs the precomputed neighbor dictionaries for A and W
            logger.info("Precomputing neighbor matrices for BFS retrieval...")
            self._precompute_bfs_neighbors()
        retrieval_results = []
        for question_info in tqdm(questions, desc="Retrieving"):
            question = question_info["question"]
            question_embedding = self.config.embedding_model.encode(
                question,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=self.config.batch_size,
            )
            (
                seed_entity_indices,
                seed_entities,
                seed_entity_hash_ids,
                seed_entity_scores,
            ) = self.get_seed_entities(question)
            if len(seed_entities) != 0:
                sorted_passage_hash_ids, sorted_passage_scores = (
                    self.graph_search_with_seed_entities(
                        question,
                        question_embedding,
                        seed_entity_indices,
                        seed_entities,
                        seed_entity_hash_ids,
                        seed_entity_scores,
                    )
                )
                final_passage_hash_ids = sorted_passage_hash_ids[
                    : self.config.retrieval_top_k
                ]
                final_passage_scores = sorted_passage_scores[
                    : self.config.retrieval_top_k
                ]
                final_passages = [
                    self.passage_embedding_store.hash_id_to_text[passage_hash_id]
                    for passage_hash_id in final_passage_hash_ids
                ]
            else:
                sorted_passage_indices, sorted_passage_scores = (
                    self.dense_passage_retrieval(question_embedding)
                )
                final_passage_indices = sorted_passage_indices[
                    : self.config.retrieval_top_k
                ]
                final_passage_scores = sorted_passage_scores[
                    : self.config.retrieval_top_k
                ]
                final_passages = [
                    self.passage_embedding_store.texts[idx]
                    for idx in final_passage_indices
                ]
            result = {
                "question": question,
                "sorted_passage": final_passages,
                "sorted_passage_scores": final_passage_scores,
                "gold_answer": question_info["answer"],
            }
            retrieval_results.append(result)
        return retrieval_results
    # =========================================================================
    # ANN acceleration: approximate nearest neighbor search replaces the O(N^2) full similarity matrix
    # =========================================================================
    @staticmethod
    def _topk_from_sparse_row(row, k):
        """Return the top-k entries (idx, val) of a 1xN scipy sparse row, in descending value order."""
        data = np.asarray(row.data, dtype=np.float32)
        idx = np.asarray(row.indices, dtype=np.int64)
        if data.size == 0:
            return idx[:0], data[:0]
        if data.size <= k:
            order = np.argsort(-data, kind="stable")
            return idx[order], data[order]
        part = np.argpartition(-data, k)[:k]
        part = part[np.argsort(-data[part], kind="stable")]
        return idx[part], data[part]
    def _compute_ann_similarity_matrix(self, k):
        """
        Build a sparse entity similarity matrix with (approximate) nearest neighbor search, replacing
        the O(N^2) full cosine computation. Each entity keeps only its k most similar neighbors (self
        excluded). This is the candidate set for the neighbor matrix fusion downstream: since
        w_ij = alpha*co_ij + (1-alpha)*sim_ij, any entity that can enter the final top-K fused
        neighbors must rank high in the co-occurrence or the similarity channel, so the final top-K
        fused neighbors are always a subset of (top-K co) union (top-K sim). Keeping only the top-K
        similarities per entity therefore reduces the similarity cost from O(N^2) to about O(N log N)
        (ANN) or O(N*k) (exact fallback), with almost no loss in the final neighbor set. Backend
        priority: faiss (HNSW/Flat) -> sklearn NearestNeighbors (brute) -> torch blockwise exact
        top-k (N x N is never materialized). Returns a scipy csr_matrix(N, N) with at most k entries per row, holding cosine similarities as float32.
        """
        num_entities = len(self.entity_hash_ids)
        emb = np.ascontiguousarray(np.asarray(self.entity_embeddings, dtype=np.float32))
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb_n = emb / norms  # L2 normalize so that the dot product equals the cosine similarity
        K = int(k) + 1  # +1 to exclude the self match
        rows, cols, data = [], [], []
        # ---- 1) faiss (approximate or exact) ----
        try:
            import faiss
            d = emb_n.shape[1]
            idx_type = getattr(self.config, "ann_index_type", "hnsw")
            if idx_type == "flat":
                index = faiss.IndexFlatIP(d)  # Exact inner product equals cosine similarity
            else:
                index = faiss.IndexHNSWFlat(d, 32)
                index.hnsw.efConstruction = int(getattr(self.config, "ann_efConstruction", 200))
                index.hnsw.efSearch = int(getattr(self.config, "ann_efSearch", 128))
            index.add(emb_n)
            D, I = index.search(emb_n, K)
            for i in range(num_entities):
                for r in range(K):
                    j = int(I[i, r])
                    if j == i or j < 0:
                        continue
                    rows.append(i)
                    cols.append(j)
                    data.append(float(D[i, r]))
            logger.info(f"Entity similarity via faiss ANN (index={idx_type}, top-k={int(k)})")
            return csr_matrix(
                (data, (rows, cols)), shape=(num_entities, num_entities), dtype=np.float32
            )
        except ImportError:
            logger.warning("faiss not available, falling back to sklearn/torch for ANN sim")
        # ---- 2) sklearn NearestNeighbors (exact brute force) ----
        try:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=K, metric="cosine", algorithm="brute")
            nn.fit(emb_n)
            dist, nbr = nn.kneighbors(emb_n, n_neighbors=K)
            sim = 1.0 - dist
            for i in range(num_entities):
                for r in range(K):
                    j = int(nbr[i, r])
                    if j == i:
                        continue
                    rows.append(i)
                    cols.append(j)
                    data.append(float(sim[i, r]))
            logger.info(f"Entity similarity via sklearn NearestNeighbors (top-k={int(k)})")
            return csr_matrix(
                (data, (rows, cols)), shape=(num_entities, num_entities), dtype=np.float32
            )
        except ImportError:
            logger.warning("sklearn not available, falling back to torch blockwise top-k")
        # ---- 3) torch blockwise exact top-k (N x N is never materialized) ----
        logger.info(f"Entity similarity via torch blockwise top-k (top-k={int(k)})")
        ten = torch.from_numpy(emb_n)
        bs = 4096
        for s in range(0, num_entities, bs):
            e = min(s + bs, num_entities)
            sim_blk = ten[s:e] @ ten.T
            for off in range(s, e):
                sim_blk[off - s, off] = -1e30  # Exclude the self match
            # The torch branch already masks the self match, so take int(k) real neighbors
            # (faiss/sklearn return the self match and need K+1 with it removed)
            topk = min(int(k), num_entities)
            vals, inds = torch.topk(sim_blk, topk, dim=1)
            vals = vals.cpu().numpy()
            inds = inds.cpu().numpy()
            for r in range(e - s):
                i = s + r
                for c in range(topk):
                    j = int(inds[r, c])
                    if j == i:
                        continue
                    rows.append(i)
                    cols.append(j)
                    data.append(float(vals[r, c]))
        return csr_matrix(
            (data, (rows, cols)), shape=(num_entities, num_entities), dtype=np.float32
        )
    def _precompute_bfs_neighbors(self):
        """
        Precompute the neighbor dictionary for the BFS mode (cached per alpha, appended incrementally).
        Only the cooccur_alpha passed in is computed and cached: a cache hit is read directly, a miss is
        computed and appended back. The co-occurrence and similarity matrices do not depend on alpha and are reused from cache; fusion and thresholding run only for the current alpha.
        """
        num_entities = len(self.entity_hash_ids)
        num_sentences = len(self.sentence_hash_ids)
        threshold = self.config.precompute_threshold
        MAX_NEIGHBORS = [self.config.top_k_sentence, self.top_k_entity_cooccur]
        self.entity_neigh_log = os.path.join(
            self.cache_dir, "entity_cooccur_neighbors.log"
        )
        current_alpha = round(self.cooccur_alpha, 2)
        # 1) Neighbor cache first: when the current alpha is already computed, read it and return
        # directly (no need to load the co-occurrence or similarity matrices)
        if os.path.exists(self.cache_entity_neigh):
            with open(self.cache_entity_neigh, "rb") as f:
                alpha2neigh = pickle.load(f)
            if current_alpha in alpha2neigh:
                neigh_list = alpha2neigh[current_alpha]
                self.entity_cooccur_neighbors = neigh_list
                total = len(neigh_list)
                cnt_zero = sum(1 for lst in neigh_list if len(lst) == 0)
                cnt_lt5 = sum(1 for lst in neigh_list if len(lst) < 5)
                cnt_lt10 = sum(1 for lst in neigh_list if len(lst) < 10)
                total_neigh = sum(len(lst) for lst in neigh_list)
                avg_neigh = total_neigh / total if total > 0 else 0.0
                zero_rate = cnt_zero / total * 100 if total > 0 else 0.0
                lt5_rate = cnt_lt5 / total * 100 if total > 0 else 0.0
                lt10_rate = cnt_lt10 / total * 100 if total > 0 else 0.0
                line = (
                    f"[Entity Neighbors(Loaded, alpha={current_alpha})] total_nodes:{total}, zero:{cnt_zero}, nodes<5:{cnt_lt5}, "
                    f"total_neighbors:{total_neigh}, avg_neighbors:{avg_neigh:.2f} | "
                    f"zero_rate:{zero_rate:.2f}%, lt5_rate:{lt5_rate:.2f}%, lt10_rate:{lt10_rate:.2f}%"
                )
                logger.info(line)
                with open(self.entity_neigh_log, "w", encoding="utf-8") as f:
                    f.write(line + "\n")
                logger.info(f"Entity statistics log saved to: {self.entity_neigh_log}")
                logger.info(
                    f"Reuse cached entity neighbors for alpha={current_alpha} from: {self.cache_entity_neigh}"
                )
                return
            # The current alpha is not cached yet: keep the existing cache and append the new entry
            # after computing it
            logger.info(
                f"alpha={current_alpha} not yet cached, will compute and append to: {self.cache_entity_neigh}"
            )
        else:
            alpha2neigh = {}
        # ===================== Load or build the sparse entity co-occurrence matrix (alpha-independent, cacheable) =====================
        if os.path.exists(self.cache_entity_cooccur):
            logger.info(
                f"Load sparse entity co-occurrence from cache: {self.cache_entity_cooccur}"
            )
            entity_cooccur = load_npz(self.cache_entity_cooccur)
        else:
            logger.info("Counting entity co-occurrence...")
            num_entities = len(self.entity_hash_ids)
            coo_dict = defaultdict(int)
            for sentence_entities in tqdm(
                self.sentence_hash_id_to_entity_hash_ids.values(),
                desc="Count entity co-occur",
            ):
                if len(sentence_entities) < 2:
                    continue
                entity_indices = [
                    self.entity_embedding_store.hash_id_to_idx[e]
                    for e in sentence_entities
                ]
                ids = np.array(entity_indices)
                i_mat, j_mat = np.meshgrid(ids, ids)
                mask = i_mat != j_mat
                for i, j in zip(i_mat[mask], j_mat[mask]):
                    coo_dict[(i, j)] += 1
            row_data = defaultdict(list)
            for (i, j), cnt in coo_dict.items():
                row_data[i].append((cnt, j))
            rows, cols, data = [], [], []
            max_per_row = 200
            ratio = 0.1
            for i, items in tqdm(row_data.items(), desc="Filter co-occur top"):
                items_sorted = sorted(items, key=lambda x: x[0], reverse=True)
                take = min(max(1, int(len(items_sorted) * ratio)), max_per_row)
                top = items_sorted[:take]
                for val, j in top:
                    rows.append(i)
                    cols.append(j)
                    data.append(val)
            entity_cooccur = csr_matrix(
                (data, (rows, cols)),
                shape=(num_entities, num_entities),
                dtype=np.uint16,
            )
            cache_dir = os.path.dirname(self.cache_entity_cooccur)
            os.makedirs(cache_dir, exist_ok=True)
            save_npz(self.cache_entity_cooccur, entity_cooccur)
            logger.info(
                f"Saved sparse co-occur cache to: {self.cache_entity_cooccur}"
            )
        # ===================== Load or build the sparse entity similarity matrix (ANN, alpha-independent) =====================
        if self.use_ann_sim and os.path.exists(self.cache_entity_sim_ann):
            logger.info(
                f"Load ANN entity similarity from cache: {self.cache_entity_sim_ann}"
            )
            entity_sim_matrix = load_npz(self.cache_entity_sim_ann)
        elif self.use_ann_sim:
            logger.info("Computing entity similarity via ANN (top-k) ...")
            k_sim = max(1, int(self.ann_topk))
            entity_sim_matrix = self._compute_ann_similarity_matrix(k_sim)
            cache_dir = os.path.dirname(self.cache_entity_sim_ann)
            os.makedirs(cache_dir, exist_ok=True)
            save_npz(self.cache_entity_sim_ann, entity_sim_matrix)
            logger.info(
                f"Saved ANN similarity cache to: {self.cache_entity_sim_ann}"
            )
        else:
            if os.path.exists(self.cache_entity_sim):
                logger.info(
                    f"Load sparse entity similarity from cache: {self.cache_entity_sim}"
                )
                entity_sim_matrix = load_npz(self.cache_entity_sim)
            else:
                logger.info("Computing entity similarity matrix (dense O(N^2))...")
                num_entities = len(self.entity_hash_ids)
                entity_embeddings_tensor = torch.from_numpy(
                    self.entity_embeddings
                ).float()
                ent_norm = torch.norm(
                    entity_embeddings_tensor, p=2, dim=1, keepdim=True
                )
                ent_norm[ent_norm == 0] = 1.0
                batch = 2000
                full_sim_np = np.zeros(
                    (num_entities, num_entities), dtype=np.float16
                )
                for s in range(0, num_entities, batch):
                    e = min(s + batch, num_entities)
                    batch_emb = entity_embeddings_tensor[s:e]
                    sim_batch = batch_emb @ entity_embeddings_tensor.T
                    sim_batch = sim_batch / (ent_norm[s:e] @ ent_norm.T)
                    full_sim_np[s:e] = sim_batch.cpu().numpy().astype(np.float16)
                np.fill_diagonal(full_sim_np, -1.0)
                rows, cols, data = [], [], []
                max_per_row = 200
                ratio = 0.1
                for i in tqdm(range(num_entities), desc="Filter sim top items fast"):
                    row = full_sim_np[i]
                    take = max(1, min(int(len(row) * ratio), max_per_row))
                    top_idx = np.argpartition(-row, take)[:take]
                    vals = row[top_idx]
                    rows.extend([i] * take)
                    cols.extend(top_idx.tolist())
                    data.extend(vals.tolist())
                entity_sim_matrix = csr_matrix(
                    (data, (rows, cols)),
                    shape=(num_entities, num_entities),
                    dtype=np.float32,
                )
                cache_dir = os.path.dirname(self.cache_entity_sim)
                os.makedirs(cache_dir, exist_ok=True)
                save_npz(self.cache_entity_sim, entity_sim_matrix)
                logger.info(
                    f"Saved sparse similarity cache to: {self.cache_entity_sim}"
                )
        # ===================== Compute the neighbors for the current alpha only and append them to the cache =====================
        logger.info(f"Building entity neighbors for alpha={current_alpha} ...")
        neighbors = [[] for _ in range(num_entities)]
        K = self.top_k_entity_cooccur
        for i in tqdm(range(num_entities), desc=f"alpha={current_alpha} build"):
            co_idx, co_val = self._topk_from_sparse_row(entity_cooccur[i], K)
            sim_idx, sim_val = self._topk_from_sparse_row(entity_sim_matrix[i], K)
            if co_idx.size == 0 and sim_idx.size == 0:
                continue
            cand = np.unique(np.concatenate([co_idx, sim_idx]))
            co_map = dict(zip(co_idx.tolist(), co_val.tolist()))
            sim_map = dict(zip(sim_idx.tolist(), sim_val.tolist()))
            co_arr = np.array(
                [co_map.get(int(j), 0.0) for j in cand], dtype=np.float32
            )
            sim_arr = np.array(
                [sim_map.get(int(j), 0.0) for j in cand], dtype=np.float32
            )
            co_max = co_arr.max()
            co_n = co_arr / co_max if co_max > 0 else co_arr
            sim_n = min_max_normalize(sim_arr)
            w_row = current_alpha * co_n + (1 - current_alpha) * sim_n
            valid_mask = w_row >= threshold
            valid_idx = cand[valid_mask]
            valid_val = w_row[valid_mask]
            if valid_idx.size == 0:
                continue
            if valid_idx.size > MAX_NEIGHBORS[1]:
                sort_idx = np.argsort(-valid_val)[: MAX_NEIGHBORS[1]]
                valid_idx = valid_idx[sort_idx]
                valid_val = valid_val[sort_idx]
            for j, v in zip(valid_idx.tolist(), valid_val.tolist()):
                neighbors[i].append((int(j), float(v)))
        alpha2neigh[current_alpha] = neighbors
        total = len(neighbors)
        cnt_zero = sum(1 for lst in neighbors if len(lst) == 0)
        cnt_lt5 = sum(1 for lst in neighbors if len(lst) < 5)
        cnt_lt10 = sum(1 for lst in neighbors if len(lst) < 10)
        total_neigh = sum(len(lst) for lst in neighbors)
        avg_neigh = total_neigh / total if total > 0 else 0.0
        zero_rate = cnt_zero / total * 100 if total > 0 else 0.0
        lt5_rate = cnt_lt5 / total * 100 if total > 0 else 0.0
        lt10_rate = cnt_lt10 / total * 100 if total > 0 else 0.0
        line = (
            f"[Entity Neighbors(Computed, alpha={current_alpha})] total_nodes:{total}, zero:{cnt_zero}, nodes<5:{cnt_lt5}, "
            f"total_neighbors:{total_neigh}, avg_neighbors:{avg_neigh:.2f} | "
            f"zero_rate:{zero_rate:.2f}%, lt5_rate:{lt5_rate:.2f}%, lt10_rate:{lt10_rate:.2f}%"
        )
        logger.info(line)
        with open(self.entity_neigh_log, "w", encoding="utf-8") as f:
            f.write(line + "\n")
        logger.info(f"Entity statistics log saved to: {self.entity_neigh_log}")
        with open(self.cache_entity_neigh, "wb") as f:
            pickle.dump(alpha2neigh, f)
        logger.info(
            f"Save entity neighbors (alpha={current_alpha}) to cache: {self.cache_entity_neigh}"
        )
        self.entity_cooccur_neighbors = alpha2neigh[current_alpha]
    def _precompute_sparse_matrices(self):
        """
        Precompute and cache sparse adjacency matrices for efficient vectorized retrieval using PyTorch.
        This is called once at the beginning of retrieve() to avoid rebuilding matrices per query.
        """
        num_entities = len(self.entity_hash_ids)
        num_sentences = len(self.sentence_hash_ids)
        threshold = self.config.precompute_threshold
        MAX_NEIGHBORS = [self.config.top_k_sentence, self.top_k_entity_cooccur]
        # Build entity-to-sentence matrix (Mention matrix) using COO format
        entity_to_sentence_indices = []
        entity_to_sentence_values = []
        for (
            entity_hash_id,
            sentence_hash_ids,
        ) in self.entity_hash_id_to_sentence_hash_ids.items():
            entity_idx = self.entity_embedding_store.hash_id_to_idx[entity_hash_id]
            for sentence_hash_id in sentence_hash_ids:
                sentence_idx = self.sentence_embedding_store.hash_id_to_idx[
                    sentence_hash_id
                ]
                entity_to_sentence_indices.append([entity_idx, sentence_idx])
                entity_to_sentence_values.append(1.0)
        # Build sentence-to-entity matrix
        sentence_to_entity_indices = []
        sentence_to_entity_values = []
        for (
            sentence_hash_id,
            entity_hash_ids,
        ) in self.sentence_hash_id_to_entity_hash_ids.items():
            sentence_idx = self.sentence_embedding_store.hash_id_to_idx[
                sentence_hash_id
            ]
            for entity_hash_id in entity_hash_ids:
                entity_idx = self.entity_embedding_store.hash_id_to_idx[entity_hash_id]
                sentence_to_entity_indices.append([sentence_idx, entity_idx])
                sentence_to_entity_values.append(1.0)
        # Convert to PyTorch sparse tensors (COO format, then convert to CSR for efficiency)
        if len(entity_to_sentence_indices) > 0:
            e2s_indices = torch.tensor(entity_to_sentence_indices, dtype=torch.long).t()
            e2s_values = torch.tensor(entity_to_sentence_values, dtype=torch.float32)
            self.entity_to_sentence_sparse = torch.sparse_coo_tensor(
                e2s_indices,
                e2s_values,
                (num_entities, num_sentences),
                device=self.device,
            ).coalesce()
        else:
            self.entity_to_sentence_sparse = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros(0, dtype=torch.float32),
                (num_entities, num_sentences),
                device=self.device,
            )
        if len(sentence_to_entity_indices) > 0:
            s2e_indices = torch.tensor(sentence_to_entity_indices, dtype=torch.long).t()
            s2e_values = torch.tensor(sentence_to_entity_values, dtype=torch.float32)
            self.sentence_to_entity_sparse = torch.sparse_coo_tensor(
                s2e_indices,
                s2e_values,
                (num_sentences, num_entities),
                device=self.device,
            ).coalesce()
        else:
            self.sentence_to_entity_sparse = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros(0, dtype=torch.float32),
                (num_sentences, num_entities),
                device=self.device,
            )
        # ==============================================
        # Sentence attention matrix A (threshold filtering only)
        # ==============================================
        logger.info("Precomputing sentence attention matrix A (threshold-only)...")
        sentence_embeddings_tensor = (
            torch.from_numpy(self.sentence_embeddings).float().to(self.device)
        )
        for i in tqdm(range(num_sentences), desc="Building sentence attention"):
            sims = (
                (sentence_embeddings_tensor[i] @ sentence_embeddings_tensor.T)
                .cpu()
                .numpy()
                .ravel()
            )
            sims = min_max_normalize(sims)
            # Threshold filtering replaces top-k
            valid_mask = sims >= threshold
            valid_idx = np.where(valid_mask)[0]
            valid_val = sims[valid_idx]
            if len(valid_val) == 0:
                continue
            if len(valid_val) > MAX_NEIGHBORS[0]:
                sort_idx = np.argsort(-valid_val)
                sort_idx = sort_idx[: MAX_NEIGHBORS[0]]
                valid_idx = valid_idx[sort_idx]
                valid_val = valid_val[sort_idx]
        # ==============================================
        # Entity weak co-occurrence weight matrix W (threshold filtering only)
        # ==============================================
        logger.info(
            "Precomputing entity weak co-occurrence matrix W (threshold-only)..."
        )
        entity_cooccur = np.zeros((num_entities, num_entities))
        for sentence_entities in self.sentence_hash_id_to_entity_hash_ids.values():
            if len(sentence_entities) < 2:
                continue
            entity_indices = [
                self.entity_embedding_store.hash_id_to_idx[e] for e in sentence_entities
            ]
            for idx_i in entity_indices:
                for idx_j in entity_indices:
                    if idx_i != idx_j:
                        entity_cooccur[idx_i, idx_j] += 1
        # Entity similarity: reuse the ANN top-k sparse matrix, avoiding O(N^2) row-wise dot products
        ann_sim = (
            self._compute_ann_similarity_matrix(max(1, int(self.ann_topk)))
            if self.use_ann_sim
            else None
        )
        entity_embeddings_tensor = None  # Used only in the fallback path
        W_indices = []
        W_values = []
        K = self.top_k_entity_cooccur
        for i in tqdm(range(num_entities), desc="Building entity co-occurrence"):
            # Co-occurrence candidates: top K of the dense co-occurrence row
            co_row = entity_cooccur[i].ravel()
            if co_row.size > K:
                co_rc = np.argpartition(-co_row, K)[:K]
            elif co_row.size > 0:
                co_rc = np.argsort(-co_row)
            else:
                co_rc = np.array([], dtype=np.int64)
            co_idx = co_rc.astype(np.int64)
            co_val = co_row[co_rc]
            # Similarity candidates: ANN top-k (falls back to row-wise dot products)
            if ann_sim is not None:
                sim_idx, sim_val = self._topk_from_sparse_row(ann_sim[i], K)
            else:
                if entity_embeddings_tensor is None:
                    entity_embeddings_tensor = (
                        torch.from_numpy(self.entity_embeddings).float().to(self.device)
                    )
                sim_vec = (
                    (entity_embeddings_tensor[i] @ entity_embeddings_tensor.T)
                    .cpu()
                    .numpy()
                    .ravel()
                )
                if sim_vec.size > K:
                    st = np.argpartition(-sim_vec, K)[:K]
                else:
                    st = np.argsort(-sim_vec)
                sim_idx = st.astype(np.int64)
                sim_val = sim_vec[st]
            if co_idx.size == 0 and sim_idx.size == 0:
                continue
            # Candidate pruning: compute w only over the union of the top-K co-occurrence and top-K similarity sets
            cand = np.unique(np.concatenate([co_idx, sim_idx]))
            co_map = dict(zip(co_idx.tolist(), co_val.tolist()))
            sim_map = dict(zip(sim_idx.tolist(), sim_val.tolist()))
            co_arr = np.array(
                [co_map.get(int(j), 0.0) for j in cand], dtype=np.float32
            )
            sim_arr = np.array(
                [sim_map.get(int(j), 0.0) for j in cand], dtype=np.float32
            )
            co_n = min_max_normalize(co_arr)
            sim_n = min_max_normalize(sim_arr)
            w_row = self.cooccur_alpha * co_n + (1 - self.cooccur_alpha) * sim_n
            # Threshold filtering replaces top-k
            valid_mask = w_row >= threshold
            valid_idx = cand[valid_mask]
            valid_val = w_row[valid_mask]
            if valid_idx.size == 0:
                continue
            if valid_idx.size > MAX_NEIGHBORS[1]:
                sort_idx = np.argsort(-valid_val)[: MAX_NEIGHBORS[1]]
                valid_idx = valid_idx[sort_idx]
                valid_val = valid_val[sort_idx]
            for j, v in zip(valid_idx.tolist(), valid_val.tolist()):
                W_indices.append([i, int(j)])
                W_values.append(float(v))
        if len(W_values) > 0:
            W_indices_t = torch.tensor(W_indices, dtype=torch.long).t()
            W_values_t = torch.tensor(W_values, dtype=torch.float32)
            self.entity_weak_cooccur_W = torch.sparse_coo_tensor(
                W_indices_t,
                W_values_t,
                (num_entities, num_entities),
                device=self.device,
            ).coalesce()
        else:
            self.entity_weak_cooccur_W = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros(0, dtype=torch.float32),
                (num_entities, num_entities),
                device=self.device,
            )
    def graph_search_with_seed_entities(
        self,
        question,
        question_embedding,
        seed_entity_indices,
        seed_entities,
        seed_entity_hash_ids,
        seed_entity_scores,
    ):
        if self.config.use_vectorized_retrieval:
            entity_weights, actived_entities = self.calculate_entity_scores_vectorized(
                question_embedding,
                seed_entity_indices,
                seed_entities,
                seed_entity_hash_ids,
                seed_entity_scores,
            )
        else:
            entity_weights, actived_entities = self.calculate_entity_scores(
                question_embedding,
                seed_entity_indices,
                seed_entities,
                seed_entity_hash_ids,
                seed_entity_scores,
            )
        passage_weights = self.calculate_passage_scores(
            question, question_embedding, actived_entities
        )
        node_weights = entity_weights + passage_weights
        ppr_sorted_passage_indices, ppr_sorted_passage_scores = self.run_ppr(
            node_weights
        )
        return ppr_sorted_passage_indices, ppr_sorted_passage_scores
    def run_ppr(self, node_weights):
        reset_prob = np.where(
            np.isnan(node_weights) | (node_weights < 0), 0, node_weights
        )
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=self.config.damping,
            directed=False,
            weights="weight",
            reset=reset_prob,
            implementation="prpack",
        )
        doc_scores = np.array(
            [pagerank_scores[idx] for idx in self.passage_node_indices]
        )
        sorted_indices_in_doc_scores = np.argsort(doc_scores)[::-1]
        sorted_passage_scores = doc_scores[sorted_indices_in_doc_scores]
        sorted_passage_hash_ids = [
            self.vertex_idx_to_node_name[self.passage_node_indices[i]]
            for i in sorted_indices_in_doc_scores
        ]
        return sorted_passage_hash_ids, sorted_passage_scores.tolist()
    def calculate_entity_scores(
        self,
        question_embedding,
        seed_entity_indices,
        seed_entities,
        seed_entity_hash_ids,
        seed_entity_scores,
    ):
        # BFS version adapted to the improved pipeline
        actived_entities = {}
        entity_weights = np.zeros(len(self.graph.vs["name"]))
        num_sentences = len(self.sentence_hash_ids)
        for seed_entity_idx, seed_entity, seed_entity_hash_id, seed_entity_score in zip(
            seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores
        ):
            actived_entities[seed_entity_hash_id] = (
                seed_entity_idx,
                seed_entity_score,
                1,
            )
            seed_entity_node_idx = self.node_name_to_vertex_idx[seed_entity_hash_id]
            entity_weights[seed_entity_node_idx] = seed_entity_score
        used_sentence_hash_ids = set()
        current_entities = actived_entities.copy()
        iteration = 1
        # ==============================================
        # Use the raw sigma_q: no early full sentence smoothing
        # ==============================================
        question_emb = (
            question_embedding.reshape(-1, 1)
            if len(question_embedding.shape) == 1
            else question_embedding
        )
        # Raw sigma_q
        raw_sentence_sims = np.dot(self.sentence_embeddings, question_emb).flatten()
        sentence_similarities = raw_sentence_sims
        while len(current_entities) > 0 and iteration < self.config.max_iterations:
            new_entities = {}
            # === Step 1: handle the sentences of the current entities first, this is the core hard propagation ===
            for entity_hash_id, (
                entity_id,
                entity_score,
                tier,
            ) in current_entities.items():
                if entity_score < self.config.iteration_threshold:
                    continue
                sentence_hash_ids = [
                    sid
                    for sid in list(
                        self.entity_hash_id_to_sentence_hash_ids[entity_hash_id]
                    )
                    if sid not in used_sentence_hash_ids
                ]
                if not sentence_hash_ids:
                    continue
                sentence_indices = [
                    self.sentence_embedding_store.hash_id_to_idx[sid]
                    for sid in sentence_hash_ids
                ]
                sentence_sims = sentence_similarities[sentence_indices]
                top_sentence_indices = np.argsort(sentence_sims)[::-1][
                    : self.config.top_k_sentence
                ]
                for top_sentence_index in top_sentence_indices:
                    top_sentence_hash_id = sentence_hash_ids[top_sentence_index]
                    top_sentence_score = sentence_sims[top_sentence_index]
                    used_sentence_hash_ids.add(top_sentence_hash_id)
                    entity_hash_ids_in_sentence = (
                        self.sentence_hash_id_to_entity_hash_ids[top_sentence_hash_id]
                    )
                    for next_entity_hash_id in entity_hash_ids_in_sentence:
                        next_entity_score = entity_score * top_sentence_score
                        if next_entity_score < self.config.iteration_threshold:
                            continue
                        # Base score
                        base_score = next_entity_score
                        # Look up the co-occurrence neighbor weight from the current entity to the target entity
                        cooccur_weight = None
                        # entity_id: index of the current entity
                        for neighbor_id, w in self.entity_cooccur_neighbors[entity_id]:
                            # Index of the target entity
                            next_entity_idx = (
                                self.entity_embedding_store.hash_id_to_idx[
                                    next_entity_hash_id
                                ]
                            )
                            if neighbor_id == next_entity_idx:
                                cooccur_weight = w
                                break
                        # Take the minimum of the two; without a co-occurrence weight, clamp to the iteration threshold
                        if cooccur_weight is not None:
                            next_entity_score = max(
                                min(base_score, cooccur_weight),
                                self.config.iteration_threshold,
                            )
                        else:
                            next_entity_score = self.config.iteration_threshold
                        next_enitity_node_idx = self.node_name_to_vertex_idx[
                            next_entity_hash_id
                        ]
                        entity_weights[next_enitity_node_idx] += next_entity_score
                        new_entities[next_entity_hash_id] = (
                            next_enitity_node_idx,
                            next_entity_score,
                            iteration + 1,
                        )
            # === Step 2: after the sentence stage, extend new_entities with the weak neighbors ===
            w_current_entities = new_entities.copy()
            for entity_hash_id, (entity_id, entity_score, tier) in new_entities.items():
                # Weak neighbors are added on top, they replace nothing
                for cooccur_entity_id, w in self.entity_cooccur_neighbors[entity_id]:
                    new_score = (
                        entity_score * w
                    )  # weak neighbors carry a low weight
                    if new_score < self.config.iteration_threshold:
                        continue
                    cooccur_entity_hash_id = self.entity_hash_ids[cooccur_entity_id]
                    if cooccur_entity_hash_id not in w_current_entities:
                        node_idx = self.node_name_to_vertex_idx[cooccur_entity_hash_id]
                        w_current_entities[cooccur_entity_hash_id] = (
                            cooccur_entity_id,
                            new_score,
                            tier,
                        )
                        entity_weights[node_idx] += new_score
            # The next round handles the new entities plus their weak neighbors
            current_entities = w_current_entities
            actived_entities.update(current_entities)
            iteration += 1
        return entity_weights, actived_entities
    def calculate_entity_scores_vectorized(
        self,
        question_embedding,
        seed_entity_indices,
        seed_entities,
        seed_entity_hash_ids,
        seed_entity_scores,
    ):
        """
        GPU-accelerated vectorized version using PyTorch sparse tensors.
        """
        entity_weights = np.zeros(len(self.graph.vs["name"]))
        num_entities = len(self.entity_hash_ids)
        num_sentences = len(self.sentence_hash_ids)
        question_emb = (
            question_embedding.reshape(-1, 1)
            if len(question_embedding.shape) == 1
            else question_embedding
        )
        sentence_similarities_np = np.dot(
            self.sentence_embeddings, question_emb
        ).flatten()
        sentence_similarities = (
            torch.from_numpy(sentence_similarities_np).float().to(self.device)
        )
        # ==============================================
        # Use the raw sigma_q: no early full sentence smoothing
        # ==============================================
        # Track used sentences for deduplication (like BFS version)
        used_sentence_mask = torch.zeros(
            num_sentences, dtype=torch.bool, device=self.device
        )
        seed_indices = torch.tensor(
            [[idx] for idx in seed_entity_indices], dtype=torch.long
        ).t()
        seed_values = torch.tensor(seed_entity_scores, dtype=torch.float32)
        entity_scores_sparse = torch.sparse_coo_tensor(
            seed_indices, seed_values, (num_entities,), device=self.device
        ).coalesce()
        # Also maintain a dense accumulator for total scores
        entity_scores_dense = torch.zeros(
            num_entities, dtype=torch.float32, device=self.device
        )
        entity_scores_dense.scatter_(
            0,
            torch.tensor(seed_entity_indices, device=self.device),
            torch.tensor(seed_entity_scores, dtype=torch.float32, device=self.device),
        )
        actived_entities = {}
        for seed_entity_idx, seed_entity, seed_entity_hash_id, seed_entity_score in zip(
            seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores
        ):
            actived_entities[seed_entity_hash_id] = (
                seed_entity_idx,
                seed_entity_score,
                0,
            )
            seed_entity_node_idx = self.node_name_to_vertex_idx[seed_entity_hash_id]
            entity_weights[seed_entity_node_idx] = seed_entity_score
        current_entity_scores_sparse = entity_scores_sparse
        # Iterative matrix-based propagation using sparse matrices on GPU
        for iteration in range(1, self.config.max_iterations):
            current_entity_scores_dense = current_entity_scores_sparse.to_dense()
            current_entity_scores_dense = torch.where(
                current_entity_scores_dense >= self.config.iteration_threshold,
                current_entity_scores_dense,
                torch.zeros_like(current_entity_scores_dense),
            )
            nonzero_mask = current_entity_scores_dense > 0
            nonzero_indices = torch.nonzero(nonzero_mask, as_tuple=False).squeeze(-1)
            if len(nonzero_indices) == 0:
                break
            nonzero_values = current_entity_scores_dense[nonzero_indices]
            current_entity_scores_sparse = torch.sparse_coo_tensor(
                nonzero_indices.unsqueeze(0),
                nonzero_values,
                (num_entities,),
                device=self.device,
            ).coalesce()
            # Step 1: Sparse entity scores @ Sparse E2S matrix
            # Convert sparse vector to 2D for matrix multiplication
            current_scores_2d = torch.sparse_coo_tensor(
                torch.stack([nonzero_indices, torch.zeros_like(nonzero_indices)]),
                nonzero_values,
                (num_entities, 1),
                device=self.device,
            ).coalesce()
            # E @ E2S -> sentence activation scores
            sentence_activation = torch.sparse.mm(
                self.entity_to_sentence_sparse.t(), current_scores_2d  # i.e. the M matrix
            )
            # Convert to dense before squeeze to avoid CUDA sparse tensor issues
            if sentence_activation.is_sparse:
                sentence_activation = sentence_activation.to_dense()
            sentence_activation = sentence_activation.squeeze()
            # Apply sentence deduplication: mask out used sentences
            sentence_activation = torch.where(
                used_sentence_mask,
                torch.zeros_like(sentence_activation),
                sentence_activation,
            )
            # Step 2: Per-entity top-k sentence selection
            selected_sentence_indices_list = []
            if len(nonzero_indices) > 0 and self.config.top_k_sentence > 0:
                # Iterate through each active entity
                for i, entity_idx in enumerate(nonzero_indices):
                    entity_score = nonzero_values[i]
                    # Get sentences connected to this entity from the sparse matrix
                    entity_row = self.entity_to_sentence_sparse[entity_idx].coalesce()
                    entity_sentence_indices = entity_row.indices()[0]
                    if len(entity_sentence_indices) == 0:
                        continue
                    sentence_mask = ~used_sentence_mask[entity_sentence_indices]
                    available_sentence_indices = entity_sentence_indices[sentence_mask]
                    if len(available_sentence_indices) == 0:
                        continue
                    sentence_sims = sentence_similarities[available_sentence_indices]
                    k = min(self.config.top_k_sentence, len(sentence_sims))
                    if k > 0:
                        top_k_values, top_k_local_indices = torch.topk(sentence_sims, k)
                        top_k_sentence_indices = available_sentence_indices[
                            top_k_local_indices
                        ]
                        selected_sentence_indices_list.append(top_k_sentence_indices)
                if len(selected_sentence_indices_list) > 0:
                    all_selected_sentences = torch.cat(selected_sentence_indices_list)
                    unique_selected_sentences = torch.unique(all_selected_sentences)
                    used_sentence_mask[unique_selected_sentences] = True
                    # Compute weighted sentence scores for propagation
                    weighted_sentence_scores = (
                        sentence_activation * sentence_similarities
                    )
                    mask = torch.zeros(
                        num_sentences, dtype=torch.bool, device=self.device
                    )
                    mask[unique_selected_sentences] = True
                    weighted_sentence_scores = torch.where(
                        mask,
                        weighted_sentence_scores,
                        torch.zeros_like(weighted_sentence_scores),
                    )
                else:
                    # No sentences selected, create zero vector
                    weighted_sentence_scores = torch.zeros(
                        num_sentences, dtype=torch.float32, device=self.device
                    )
            else:
                # No active entities or top_k_sentence is 0
                weighted_sentence_scores = torch.zeros(
                    num_sentences, dtype=torch.float32, device=self.device
                )
            # Step 3: Weighted sentences @ S2E -> propagate to next entities
            weighted_nonzero_mask = weighted_sentence_scores > 0
            weighted_nonzero_indices = torch.nonzero(
                weighted_nonzero_mask, as_tuple=False
            ).squeeze(-1)
            if len(weighted_nonzero_indices) > 0:
                weighted_nonzero_values = weighted_sentence_scores[
                    weighted_nonzero_indices
                ]
                weighted_scores_2d = torch.sparse_coo_tensor(
                    torch.stack(
                        [
                            weighted_nonzero_indices,
                            torch.zeros_like(weighted_nonzero_indices),
                        ]
                    ),
                    weighted_nonzero_values,
                    (num_sentences, 1),
                    device=self.device,
                ).coalesce()
                next_entity_scores_result = torch.sparse.mm(
                    self.sentence_to_entity_sparse.t(), weighted_scores_2d  # i.e. the M^T matrix
                )
                if next_entity_scores_result.is_sparse:
                    next_entity_scores_result = next_entity_scores_result.to_dense()
                next_entity_scores_dense = next_entity_scores_result.squeeze()
            else:
                next_entity_scores_dense = torch.zeros(
                    num_entities, dtype=torch.float32, device=self.device
                )
            # === Apply the W propagation to the new entity scores here, not at the beginning ===
            next_entity_scores_dense = (
                torch.sparse.mm(
                    self.entity_weak_cooccur_W, next_entity_scores_dense.unsqueeze(1)
                ).squeeze()
            )  # weak neighbors carry a low weight
            entity_scores_dense += next_entity_scores_dense
            next_entity_scores_np = next_entity_scores_dense.cpu().numpy()
            active_indices = np.where(
                next_entity_scores_np >= self.config.iteration_threshold
            )[0]
            for entity_idx in active_indices:
                score = next_entity_scores_np[entity_idx]
                entity_hash_id = self.entity_hash_ids[entity_idx]
                actived_entities[entity_hash_id] = (entity_idx, float(score), iteration)
            next_nonzero_mask = next_entity_scores_dense > 0
            next_nonzero_indices = torch.nonzero(
                next_nonzero_mask, as_tuple=False
            ).squeeze(-1)
            if len(next_nonzero_indices) > 0:
                next_nonzero_values = next_entity_scores_dense[next_nonzero_indices]
                current_entity_scores_sparse = torch.sparse_coo_tensor(
                    next_nonzero_indices,
                    next_nonzero_values,
                    (num_entities,),
                    device=self.device,
                ).coalesce()
            else:
                break
        entity_scores_final = entity_scores_dense.cpu().numpy()
        nonzero_indices = np.where(entity_scores_final > 0)[0]
        for entity_idx in nonzero_indices:
            score = entity_scores_final[entity_idx]
            entity_hash_id = self.entity_hash_ids[entity_idx]
            entity_node_idx = self.node_name_to_vertex_idx[entity_hash_id]
            entity_weights[entity_node_idx] = float(score)
        return entity_weights, actived_entities
    def calculate_passage_scores(self, question, question_embedding, actived_entities):
        passage_weights = np.zeros(len(self.graph.vs["name"]))
        dpr_passage_indices, dpr_passage_scores = self.dense_passage_retrieval(
            question_embedding
        )
        dpr_passage_scores = min_max_normalize(dpr_passage_scores)
        apply_attribute_boost = (
            self.config.enable_hybrid_attribute_fallback
            and self._is_attribute_query(question)
        )
        question_lower = question.lower()
        for i, dpr_passage_index in enumerate(dpr_passage_indices):
            total_entity_bonus = 0
            passage_hash_id = self.passage_hash_ids[dpr_passage_index]
            dpr_passage_score = dpr_passage_scores[i]
            passage_text_lower = self.passage_embedding_store.hash_id_to_text[
                passage_hash_id
            ].lower()
            for entity_hash_id, (
                entity_id,
                entity_score,
                tier,
            ) in actived_entities.items():
                entity_lower = self.entity_embedding_store.hash_id_to_text[
                    entity_hash_id
                ].lower()
                entity_occurrences = passage_text_lower.count(entity_lower)
                if entity_occurrences > 0:
                    denom = tier if tier >= 1 else 1
                    entity_bonus = (
                        entity_score * math.log(1 + entity_occurrences) / denom
                    )
                    total_entity_bonus += entity_bonus
            passage_score = np.exp(dpr_passage_score) + total_entity_bonus
            if apply_attribute_boost:
                overlap = self._attribute_keyword_overlap(
                    question_lower, passage_text_lower
                )
                if overlap > 0:
                    passage_score += self.config.attribute_keyword_boost * math.log(
                        1 + overlap
                    )
            passage_node_idx = self.node_name_to_vertex_idx[passage_hash_id]
            passage_weights[passage_node_idx] = (
                passage_score * self.config.passage_node_weight
            )
        return passage_weights
    def dense_passage_retrieval(self, question_embedding):
        question_emb = question_embedding.reshape(1, -1)
        question_passage_similarities = np.dot(
            self.passage_embeddings, question_emb.T
        ).flatten()
        sorted_passage_indices = np.argsort(question_passage_similarities)[::-1]
        sorted_passage_scores = question_passage_similarities[
            sorted_passage_indices
        ].tolist()
        return sorted_passage_indices, sorted_passage_scores
    def _is_attribute_query(self, question):
        tokens = set(re.findall(r"\w+", question.lower()))
        return any(
            keyword in tokens for keyword in self.config.attribute_query_keywords
        )
    def _attribute_keyword_overlap(self, question_lower, passage_text_lower):
        overlap = 0
        for keyword in self.config.attribute_query_keywords:
            if keyword in question_lower and keyword in passage_text_lower:
                overlap += 1
        return overlap
    def get_seed_entities(self, question):
        question_entities = list(self.spacy_ner.question_ner(question))
        if len(question_entities) == 0:
            return [], [], [], []
        question_entity_embeddings = self.config.embedding_model.encode(
            question_entities,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=self.config.batch_size,
        )
        similarities = np.dot(self.entity_embeddings, question_entity_embeddings.T)
        seed_entity_indices = []
        seed_entity_texts = []
        seed_entity_hash_ids = []
        seed_entity_scores = []
        for query_entity_idx in range(len(question_entities)):
            entity_scores = similarities[:, query_entity_idx]
            best_entity_idx = np.argmax(entity_scores)
            best_entity_score = entity_scores[best_entity_idx]
            best_entity_hash_id = self.entity_hash_ids[best_entity_idx]
            best_entity_text = self.entity_embedding_store.hash_id_to_text[
                best_entity_hash_id
            ]
            seed_entity_indices.append(best_entity_idx)
            seed_entity_texts.append(best_entity_text)
            seed_entity_hash_ids.append(best_entity_hash_id)
            seed_entity_scores.append(best_entity_score)
        return (
            seed_entity_indices,
            seed_entity_texts,
            seed_entity_hash_ids,
            seed_entity_scores,
        )
    def index(self, passages):
        self.node_to_node_stats = defaultdict(dict)
        self.entity_to_sentence_stats = defaultdict(dict)
        self.passage_embedding_store.insert_text(passages)
        hash_id_to_passage = self.passage_embedding_store.get_hash_id_to_text()
        (
            existing_passage_hash_id_to_entities,
            existing_sentence_to_entities,
            new_passage_hash_ids,
        ) = self.load_existing_data(hash_id_to_passage.keys())
        if len(new_passage_hash_ids) > 0:
            new_hash_id_to_passage = {
                k: hash_id_to_passage[k] for k in new_passage_hash_ids
            }
            new_passage_hash_id_to_entities, new_sentence_to_entities = (
                self.spacy_ner.batch_ner(
                    new_hash_id_to_passage, self.config.max_workers
                )
            )
            self.merge_ner_results(
                existing_passage_hash_id_to_entities,
                existing_sentence_to_entities,
                new_passage_hash_id_to_entities,
                new_sentence_to_entities,
            )
        self.save_ner_results(
            existing_passage_hash_id_to_entities, existing_sentence_to_entities
        )
        (
            entity_nodes,
            sentence_nodes,
            passage_hash_id_to_entities,
            self.entity_to_sentence,
            self.sentence_to_entity,
        ) = self.extract_nodes_and_edges(
            existing_passage_hash_id_to_entities, existing_sentence_to_entities
        )
        self.sentence_embedding_store.insert_text(list(sentence_nodes))
        self.entity_embedding_store.insert_text(list(entity_nodes))
        self.entity_hash_id_to_sentence_hash_ids = {}
        for entity, sentence in self.entity_to_sentence.items():
            entity_hash_id = self.entity_embedding_store.text_to_hash_id[entity]
            self.entity_hash_id_to_sentence_hash_ids[entity_hash_id] = [
                self.sentence_embedding_store.text_to_hash_id[s] for s in sentence
            ]
        self.sentence_hash_id_to_entity_hash_ids = {}
        for sentence, entities in self.sentence_to_entity.items():
            sentence_hash_id = self.sentence_embedding_store.text_to_hash_id[sentence]
            self.sentence_hash_id_to_entity_hash_ids[sentence_hash_id] = [
                self.entity_embedding_store.text_to_hash_id[e] for e in entities
            ]
        self.add_entity_to_passage_edges(passage_hash_id_to_entities)
        self.add_adjacent_passage_edges()
        self.augment_graph()
        output_graphml_path = os.path.join(
            self.config.working_dir, "v1", self.dataset_name, "NexusRAG.graphml"
        )
        os.makedirs(os.path.dirname(output_graphml_path), exist_ok=True)
        self.graph.write_graphml(output_graphml_path)
    def add_adjacent_passage_edges(self):
        passage_id_to_text = self.passage_embedding_store.get_hash_id_to_text()
        index_pattern = re.compile(r"^(\d+):")
        indexed_items = [
            (int(match.group(1)), node_key)
            for node_key, text in passage_id_to_text.items()
            if (match := index_pattern.match(text.strip()))
        ]
        indexed_items.sort(key=lambda x: x[0])
        for i in range(len(indexed_items) - 1):
            current_node = indexed_items[i][1]
            next_node = indexed_items[i + 1][1]
            self.node_to_node_stats[current_node][next_node] = 1.0
    def augment_graph(self):
        self.add_nodes()
        self.add_edges()
    def add_nodes(self):
        existing_nodes = {
            v["name"]: v for v in self.graph.vs if "name" in v.attributes()
        }
        entity_hash_id_to_text = self.entity_embedding_store.get_hash_id_to_text()
        passage_hash_id_to_text = self.passage_embedding_store.get_hash_id_to_text()
        all_hash_id_to_text = {**entity_hash_id_to_text, **passage_hash_id_to_text}
        passage_hash_ids = set(passage_hash_id_to_text)
        for hash_id, text in all_hash_id_to_text.items():
            if hash_id not in existing_nodes:
                self.graph.add_vertex(name=hash_id, content=text)
        self.node_name_to_vertex_idx = {
            v["name"]: v.index for v in self.graph.vs if "name" in v.attributes()
        }
        self.passage_node_indices = [
            self.node_name_to_vertex_idx[passage_id]
            for passage_id in passage_hash_ids
            if passage_id in self.node_name_to_vertex_idx
        ]
    def add_edges(self):
        edges = []
        weights = []
        for node_hash_id, node_to_node_stats in self.node_to_node_stats.items():
            for neighbor_hash_id, weight in node_to_node_stats.items():
                if node_hash_id == neighbor_hash_id:
                    continue
                edges.append((node_hash_id, neighbor_hash_id))
                weights.append(weight)
        self.graph.add_edges(edges)
        self.graph.es["weight"] = weights
    def add_entity_to_passage_edges(self, passage_hash_id_to_entities):
        passage_to_entity_count = {}
        passage_to_all_score = defaultdict(int)
        for passage_hash_id, entities in passage_hash_id_to_entities.items():
            passage = self.passage_embedding_store.hash_id_to_text[passage_hash_id]
            for entity in entities:
                entity_hash_id = self.entity_embedding_store.text_to_hash_id[entity]
                count = passage.count(entity)
                passage_to_entity_count[(passage_hash_id, entity_hash_id)] = count
                passage_to_all_score[passage_hash_id] += count
        for (passage_hash_id, entity_hash_id), count in passage_to_entity_count.items():
            score = count / passage_to_all_score[passage_hash_id]
            self.node_to_node_stats[passage_hash_id][entity_hash_id] = score
    def extract_nodes_and_edges(
        self, existing_passage_hash_id_to_entities, existing_sentence_to_entities
    ):
        entity_nodes = set()
        sentence_nodes = set()
        passage_hash_id_to_entities = defaultdict(set)
        entity_to_sentence = defaultdict(set)
        sentence_to_entity = defaultdict(set)
        for passage_hash_id, entities in existing_passage_hash_id_to_entities.items():
            for entity in entities:
                entity_nodes.add(entity)
                passage_hash_id_to_entities[passage_hash_id].add(entity)
        for sentence, entities in existing_sentence_to_entities.items():
            sentence_nodes.add(sentence)
            for entity in entities:
                entity_to_sentence[entity].add(sentence)
                sentence_to_entity[sentence].add(entity)
        return (
            entity_nodes,
            sentence_nodes,
            passage_hash_id_to_entities,
            entity_to_sentence,
            sentence_to_entity,
        )
    def merge_ner_results(
        self,
        existing_passage_hash_id_to_entities,
        existing_sentence_to_entities,
        new_passage_hash_id_to_entities,
        new_sentence_to_entities,
    ):
        existing_passage_hash_id_to_entities.update(new_passage_hash_id_to_entities)
        existing_sentence_to_entities.update(new_sentence_to_entities)
        return existing_passage_hash_id_to_entities, existing_sentence_to_entities
    def save_ner_results(
        self, existing_passage_hash_id_to_entities, existing_sentence_to_entities
    ):
        with open(self.ner_results_path, "w") as f:
            json.dump(
                {
                    "passage_hash_id_to_entities": existing_passage_hash_id_to_entities,
                    "sentence_to_entities": existing_sentence_to_entities,
                },
                f,
            )

