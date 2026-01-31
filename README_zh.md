# MenRouterBench
本仓库是 **MenRouterBench** 的官方实现代码库，用于评测多模型路由（Router）在多任务、多模型设置下的表现。

---

## 📦 安装与依赖

请在项目根目录下执行以下命令以安装依赖：

```bash
pip install -r requirements.txt
```

---

## 🖼️ 数据准备
请从 **（预留）** 下载所需的图像数据，并将其放置在如下目录中：

```latex
./images
```

确保最终的目录结构满足评测脚本对图像路径的要求。

---

## 🧠 VLM Server 说明
本库 **仅实现了一个基于 OpenRouter API 的 VLM Server**，用于从 OpenRouter 获取 VLM 模型的回答。

如果你希望使用 **你自己的 VLM（例如本地模型或其他 API）**，你可以：

1. 打开文件：

```latex
menbench/server/agent_api_backend.py
```

2. 注册并实现你自己的 VLM 调用类
3. **注意：务必对齐接口定义**
4. 完成后，即可像使用我们提供的 `evaluator` 类一样进行测评

---

## 🔀 Router Server 说明
在文件：

```latex
menbench/server/router_dummy_backend.py
```

中，我们已经实现了 4 种基础的 Dummy Router：

+ `MinRouterServer`
+ `MaxRouterServer`
+ `RandomRouterServer`
+ `OracleRouterServer`

你可以：

1. 注册并实现你自己的 Router
2. 对齐接口
3. 使用我们提供的 `evaluator` 直接进行评测

---

## ✅ Task 1 评测方法
你可以使用如下命令进行 **Task 1** 的评测：

```bash
python eval_task1.py \
  --subset_path configs/subset_w3/ \
  --image_root images/ \
  --model qwen/qwen3-vl-32b-instruct \
  --api_key your-api-key
```

### 配置说明
+ 一些 **不经常修改的参数** 已放置在：

```latex
configs/base_task1.yaml
```

+ 你可以直接修改该文件
+ 也可以通过 **命令行参数** 临时覆盖这些设置

---

## ✅ Task 2 评测方法
你可以使用如下命令进行 **Task 2** 的评测：

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

### Task 2 配置说明
+ 不经常修改的参数位于：

```latex
configs/base_task2.yaml
```

在 `base_task2.yaml` 中，我们预先设置了：

```yaml
router_servers:
  - "MinRouterServer"
  - "MaxRouterServer"
  - "RandomRouterServer"

compared_to_router_server: "OracleRouterServer"
```

⚠️ **注意**：

+ 这些 `RouterServer` 的名字需要你提前通过 `register` 机制进行注册
+ 注册完成后即可在配置文件中正常引用
+ 通过该机制，你可以 **批量验证不同 Router 的效果**

---

## 🚀 批量评测
你可以通过修改以下脚本来进行批量测评：

+ `evaluate_task1.sh`
+ `evaluate_task2.sh`

根据你的需求调整其中的参数组合即可。

---

## 📌 特性总结
+ 本库提供：
    - 标准化的 VLM Server 接口
    - 可扩展的 Router Server 机制
    - 统一的 Evaluator 评测流程
+ 你可以自由扩展：
    - VLM 后端
    - Router 策略
+ 所有组件只需 **接口对齐即可无缝接入评测体系**

欢迎基于 MenRouterBench 进行更多有趣的路由与多模型研究 🚀

