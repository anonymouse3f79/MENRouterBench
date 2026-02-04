# MenRouterBench

🌐 **Language / 语言**  
[English](README.md) | [中文](README_zh.md)

---

This repository is the **official implementation of MenRouterBench**, a benchmark designed to evaluate **multi-model routing (Router)** performance under **multi-task and multi-model** settings.

---

## 📦 Installation & Dependencies

Please run the following command in the project root directory to install all dependencies:

```bash
pip install -r requirements.txt
````

---

## 🖼️ Data Preparation

Please download the required data from **https://huggingface.co/datasets/anonymous10256sxwzd2/MenRouterBench**  and follow the instruction there.

---

## 🧠 VLM Server Description

This repository **only provides a VLM Server implementation based on the OpenRouter API**, which is used to obtain responses from VLM models via OpenRouter.

If you want to use **your own VLM backend** (e.g., a local model or another API), you can:

1. Open the file:

```latex
menbench/server/agent_api_backend.py
```

2. Register and implement your own VLM calling class
3. **Important: Make sure the interface is strictly aligned**
4. After that, you can evaluate your VLM using the provided `evaluator` class just like the built-in implementation

---

## 🔀 Router Server Description

In the following file:

```latex
menbench/server/router_dummy_backend.py
```

we provide **four basic Dummy Router implementations**:

* `MinRouterServer`
* `MaxRouterServer`
* `RandomRouterServer`
* `OracleRouterServer`

You can:

1. Register and implement your own Router
2. Align the interface definition
3. Directly evaluate it using the provided `evaluator`

---

## ✅ Task 1 Evaluation

You can run **Task 1 evaluation** using the following command:

```bash
python eval_task1.py \
  --subset_path configs/subset_w3/ \
  --image_root images/ \
  --model qwen/qwen3-vl-32b-instruct \
  --api_key your-api-key
```

### Configuration Details

* Some **rarely modified parameters** are defined in:

```latex
configs/base_task1.yaml
```

* You can modify this file directly
* Or temporarily override these settings via **command-line arguments**

---

## ✅ Task 2 Evaluation

You can run **Task 2 evaluation** using the following command:

```bash
python eval_task2.py \
  --wk w3 \
  --models \
    qwen_qwen3-vl-8b-instruct \
    qwen_qwen3-vl-30b-a3b-instruct \
    qwen_qwen3-vl-235b-a22b-instruct \
    qwen_qwen3-vl-32b-instruct \
  --min_model qwen_qwen3-vl-8b-instruct \
  --max_model qwen_qwen3-vl-235b-a22b-instruct \
  --switch_only \
  --group_name qwen
```

### Task 2 Configuration Details

* Parameters that are rarely modified are located in:

```latex
configs/base_task2.yaml
```

In `base_task2.yaml`, we predefine:

```yaml
router_servers:
  - "MinRouterServer"
  - "MaxRouterServer"
  - "RandomRouterServer"

compared_to_router_server: "OracleRouterServer"
```

⚠️ **Important Notes**:

* These `RouterServer` names must be registered in advance via the `register` mechanism
* Once registered, they can be directly referenced in the configuration file
* This mechanism allows you to **batch-evaluate and compare different Router strategies**

---

## 🚀 Batch Evaluation

You can perform batch evaluations by modifying the following scripts:

* `evaluate_task1.sh`
* `evaluate_task2.sh`

Simply adjust the parameter combinations according to your needs.

---

## 📌 Features Summary

* This benchmark provides:

  * A standardized VLM Server interface
  * An extensible Router Server mechanism
  * A unified Evaluator pipeline
* You can freely extend:

  * VLM backends
  * Router strategies
* All components can be seamlessly integrated into the evaluation framework **as long as the interfaces are aligned**

We welcome more exciting research on routing and multi-model systems based on **MenRouterBench** 🚀