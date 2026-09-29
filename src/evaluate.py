import json
import os
from src.utils import normalize_answer
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)
FAIL_FLAG = ["LLM request failed (retry 3/3)", "error_answer"]


class Evaluator:
    def __init__(self, llm_model, predictions_path):
        """
        :param llm_model: externally provided LLM_Model instance (dedicated evaluation model)
        :param predictions_path: path to the prediction result file
        """
        self.llm_model = llm_model
        self.predictions_path = predictions_path
        self.prediction_results = self.load_predictions()

    def load_predictions(self):
        prediction_results = json.load(
            open(self.predictions_path, "r", encoding="utf-8")
        )
        return prediction_results

    def calculate_llm_accuracy(self, question, pre_answer, gold_ans):
        system_prompt = """You are an expert evaluator.
        """
        user_prompt = f"""Please evaluate if the generated answer is correct by comparing it with the gold answer.
        Generated answer: {pre_answer}
        Gold answer: {gold_ans}

        The generated answer should be considered correct if it:
        1. Contains the key information from the gold answer
        2. Is factually accurate and consistent with the gold answer
        3. Does not contain any contradicting information

        Respond with ONLY 'correct' or 'incorrect'.
        Response:
        """

        response = self.llm_model.infer(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        resp_lower = response.strip().lower()
        if resp_lower == "correct":
            return 1.0, resp_lower
        else:
            if resp_lower != "incorrect":
                resp_lower = f"error_answer:{resp_lower}"
                print(f"\n {resp_lower} \n")
            return 0.0, resp_lower

    def re_run_failed_judge(self, fail_index_list, max_workers):
        task_list = []
        for idx in fail_index_list:
            item = self.prediction_results[idx]
            question = item["question"]
            pre = item["pred_answer"]
            gold = item["gold_answer"]
            sys_prompt = "You are an expert evaluator."
            user_prompt = f"""Please evaluate if the generated answer is correct by comparing it with the gold answer.
                Generated answer: {pre}
                Gold answer: {gold}
                The generated answer should be considered correct if it:
                1. Contains the key information from the gold answer
                2. Is factually accurate and consistent with the gold answer
                3. Does not contain any contradicting information
                Respond with ONLY 'correct' or 'incorrect'.
                Response:"""
            
            msg = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ]
            task_list.append((idx, msg))
        idx_list = [x[0] for x in task_list]
        msg_list = [x[1] for x in task_list]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            judge_res = list(
                tqdm(
                    executor.map(self.llm_model.infer, msg_list),
                    total=len(msg_list),
                    desc="Re-running failed judge samples",
                )
            )
        # Update the judge results
        for i, data_idx in enumerate(idx_list):
            out = judge_res[i]
            acc = 1.0 if out.strip().lower() == "correct" else 0.0
            self.prediction_results[data_idx]["llm_accuracy"] = acc
            self.prediction_results[data_idx]["llm_judge_answer"] = out

    def calculate_contain(self, pre_answers, gold_ans):
        if (
            pre_answers is None
            or pre_answers == ""
            or (isinstance(pre_answers, str) and pre_answers.strip() == "")
        ):
            return 0
        if (
            gold_ans is None
            or gold_ans == ""
            or (isinstance(gold_ans, str) and gold_ans.strip() == "")
        ):
            return 0
        s1 = normalize_answer(pre_answers)
        s2 = normalize_answer(gold_ans)
        if s2 in s1:
            return 1
        else:
            return 0

    def evaluate_sig_sample(self, idx, prediction):
        question = prediction.get("question", "")
        pre_answer = prediction["pred_answer"]
        gold_ans = prediction["gold_answer"]
        llm_acc, model_ans = self.calculate_llm_accuracy(question, pre_answer, gold_ans)
        contain_acc = self.calculate_contain(pre_answer, gold_ans)
        return idx, llm_acc, contain_acc, model_ans

    def evaluate(self, max_workers):
        llm_scores = [0.0] * len(self.prediction_results)
        contain_scores = [0.0] * len(self.prediction_results)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.evaluate_sig_sample, idx, pred): idx
                for idx, pred in enumerate(self.prediction_results)
            }

            completed = 0
            total_llm_score = 0.0
            total_contain_score = 0.0
            pbar = tqdm(total=len(futures), desc="Evaluating samples", unit="sample")
            for future in as_completed(futures):
                idx, llm_acc, contain_acc, model_ans = future.result()
                llm_scores[idx] = llm_acc
                contain_scores[idx] = contain_acc
                self.prediction_results[idx]["llm_judge_answer"] = model_ans
                self.prediction_results[idx]["llm_accuracy"] = llm_acc
                self.prediction_results[idx]["contain_accuracy"] = contain_acc
                total_llm_score += llm_acc
                total_contain_score += contain_acc
                completed += 1
                current_llm_acc = total_llm_score / completed
                current_contain_acc = total_contain_score / completed
                pbar.set_postfix(
                    {
                        "LLM_Acc": f"{current_llm_acc:.3f}",
                        "Contain_Acc": f"{current_contain_acc:.3f}",
                    }
                )
                pbar.update(1)
            pbar.close()

        temp_data = self.prediction_results.copy()

        temp = 0
        while temp < 10:
            temp += 1
            fail_idx = [
                i
                for i, item in enumerate(temp_data)
                if "llm_judge_answer" in item
                and any(flag in item["llm_judge_answer"] for flag in FAIL_FLAG)
            ]
            if not fail_idx:
                break
            logger.warning(
                f"Judge stage: {len(fail_idx)} failed samples detected, re-running the batch"
            )
            self.need_write = True
            self.prediction_results = temp_data
            self.re_run_failed_judge(fail_idx, max_workers)
            temp_data = self.prediction_results.copy()

        total_llm = 0.0
        total_contain = 0.0
        total_num = len(self.prediction_results)
        for item in self.prediction_results:
            total_llm += item["llm_accuracy"]
            total_contain += item["contain_accuracy"]

        llm_accuracy = (total_llm / total_num) * 100
        contain_accuracy = (total_contain / total_num) * 100

        logger.info(f"Evaluation Results:")
        logger.info(
            f"  LLM Accuracy: {llm_accuracy:.2f}% ({sum(llm_scores)}/{len(llm_scores)})"
        )
        logger.info(
            f"  Contain Accuracy: {contain_accuracy:.2f}% ({sum(contain_scores)}/{len(contain_scores)})"
        )

        with open(self.predictions_path, "w", encoding="utf-8") as f:
            json.dump(self.prediction_results, f, ensure_ascii=False, indent=4)

        eval_result_path = os.path.join(
            os.path.dirname(self.predictions_path), "evaluation_results.json"
        )
        with open(eval_result_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "llm_accuracy": f"{llm_accuracy}%",
                    "contain_accuracy": f"{contain_accuracy}%",
                },
                f,
                ensure_ascii=False,
                indent=4,
            )

        return llm_accuracy, contain_accuracy

    # Resume-aware evaluation function (independent of evaluate())
    def evaluate_with_resume(self, max_workers, save_every=100):
        """
        Resume-aware evaluation:
        1. Locate the entries missing llm_accuracy/contain_accuracy
        2. Evaluate only those entries concurrently, persisting predictions.json every save_every entries
        3. Retry the failed judge calls, compute the final metrics and write evaluation_results.json
        """
        pending_indices = [
            idx
            for idx, item in enumerate(self.prediction_results)
            if "llm_accuracy" not in item or "contain_accuracy" not in item
        ]

        if not pending_indices:
            print("[Resume-Eval] All entries are already evaluated, skipping the evaluation stage")
        else:
            print(
                f"[Resume-Eval] Entries to evaluate this round: {len(pending_indices)} / {len(self.prediction_results)}"
            )
            completed = 0
            total_llm_score = 0.0
            total_contain_score = 0.0
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self.evaluate_sig_sample, idx, self.prediction_results[idx]
                    ): idx
                    for idx in pending_indices
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(pending_indices),
                    desc="Evaluating (Resume)",
                ):
                    idx, llm_acc, contain_acc, model_ans = future.result()
                    self.prediction_results[idx]["llm_judge_answer"] = model_ans
                    self.prediction_results[idx]["llm_accuracy"] = llm_acc
                    self.prediction_results[idx]["contain_accuracy"] = contain_acc
                    completed += 1
                    total_llm_score += llm_acc
                    total_contain_score += contain_acc
                    if completed % save_every == 0:
                        with open(
                            self.predictions_path, "w", encoding="utf-8"
                        ) as f:
                            json.dump(
                                self.prediction_results,
                                f,
                                ensure_ascii=False,
                                indent=4,
                            )
                        cur_llm = total_llm_score / completed
                        cur_contain = total_contain_score / completed
                        print(
                            f"[Resume-Eval] Incremental save: {completed}/{len(pending_indices)} "
                            f"(LLM_Acc={cur_llm:.3f}, Contain_Acc={cur_contain:.3f})"
                        )

            # Final save after this round completes
            with open(self.predictions_path, "w", encoding="utf-8") as f:
                json.dump(self.prediction_results, f, ensure_ascii=False, indent=4)
            print(f"[Resume-Eval] Evaluation stage finished, {completed} entries evaluated and saved")

        # ---- Retry the failed judge calls (equivalent to the original logic) ----
        temp = 0
        while temp < 10:
            temp += 1
            fail_idx = [
                i
                for i, item in enumerate(self.prediction_results)
                if "llm_judge_answer" in item
                and any(flag in item["llm_judge_answer"] for flag in FAIL_FLAG)
            ]
            if not fail_idx:
                break
            logger.warning(
                f"[Resume-Eval] Judge stage: {len(fail_idx)} failed samples detected, re-running the batch"
            )
            self.re_run_failed_judge(fail_idx, max_workers)
            with open(self.predictions_path, "w", encoding="utf-8") as f:
                json.dump(self.prediction_results, f, ensure_ascii=False, indent=4)

        # ---- Compute the final metrics and persist them ----
        total_llm = 0.0
        total_contain = 0.0
        total_num = len(self.prediction_results)
        for item in self.prediction_results:
            total_llm += item["llm_accuracy"]
            total_contain += item["contain_accuracy"]

        llm_accuracy = (total_llm / total_num) * 100
        contain_accuracy = (total_contain / total_num) * 100

        logger.info(f"Evaluation Results (Resume):")
        logger.info(f"  LLM Accuracy: {llm_accuracy:.2f}%")
        logger.info(f"  Contain Accuracy: {contain_accuracy:.2f}%")

        with open(self.predictions_path, "w", encoding="utf-8") as f:
            json.dump(self.prediction_results, f, ensure_ascii=False, indent=4)

        eval_result_path = os.path.join(
            os.path.dirname(self.predictions_path), "evaluation_results.json"
        )
        with open(eval_result_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "llm_accuracy": f"{llm_accuracy}%",
                    "contain_accuracy": f"{contain_accuracy}%",
                },
                f,
                ensure_ascii=False,
                indent=4,
            )

        return llm_accuracy, contain_accuracy
