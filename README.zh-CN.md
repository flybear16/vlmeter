# vlmeter — 多模态大模型准确率 × 成本计量流水线

[English](README.md) | 简体中文

用自己的图片 + 标准答案，横向对比多个多模态模型（API 商用模型 + 本地开源模型）的准确率，自动出对比报告。

```bash
python3 -m vlmeter run --models deepseek-flash,gpt-4o,qwen2.5-vl-7b --data data/sample.jsonl --judge deepseek-v4-pro
```

**特性**

- 一套数据同时评多模型：OpenAI 兼容接口（GPT-4o / DeepSeek / GLM-4V / Qwen-VL / vLLM / sglang / LM Studio）、Anthropic Claude、Google Gemini、Ollama 本地模型，加模型只需在 `vlmeter/config/models.yaml` 加几行配置
- 三类任务：`classification`（分类/打标，选项匹配）、`checklist`（**一图多检查项**，如"有自行车吗+收银台干净吗"，单次调用返回全部判定）、`vqa`（开放问答，exact / contains / numeric / LLM 裁判四种判分）
- 断点续跑：结果按题缓存，中断后重跑只补缺失项；失败请求自动重试
- **成本核算**：按模型单价 + 实际 token 用量估算，报告内置"成本与性能选型权衡表"（总成本/单次成本/每千次调用成本，准确率与成本并排对比）
- 报告：总准确率、分任务、**分检查项**、分类目、平均延迟、错误样本明细，同时产出 Markdown + 自包含 HTML 两种报告

## 安装

```bash
pip install -r requirements.txt          # 本机已装可直接跳过
cp .env.example .env                     # 填入你有的 API 密钥（run 会自动加载）
```

## 快速上手（10 分钟跑通）

```bash
python3 scripts/make_sample_data.py      # 生成 15 题合成示例数据（形状/颜色/计数/OCR）
python3 -m vlmeter list-models             # 看当前可用的模型
python3 -m vlmeter run --models mock --data data/sample.jsonl        # 离线自测，不花钱
python3 -m vlmeter run --models deepseek-flash --data data/sample.jsonl --judge deepseek-v4-pro
python3 -m vlmeter report --data data/sample.jsonl                    # 重新出报告
```

报告同时输出 Markdown（`results/report_*.md`）和自包含 HTML（`results/report_*.html`，单文件无外部依赖，可直接发给别人双击打开），原始逐题结果在 `results/<模型名>/*.jsonl`。

## 接入你自己的数据

### 场景检查模式（推荐）：一张图判断多种场景

典型需求：传一张图片，判断"有没有自行车"、"收银台干不干净"等多个是/否检查项。
一行数据 = 一张图 + 该图的全部检查项，**一次模型调用**返回所有判定（省 N-1 次调用）：

```json
{"id": "shop001", "image": "images/shop001.jpg", "task": "checklist",
 "checks": [
   {"name": "图中有自行车", "answer": true},
   {"name": "收银台干净整洁", "answer": false,
    "criteria": "台面无散落垃圾、无明显污渍、无杂物堆放，轻微灰尘不算"},
   {"name": "店内无杂物堆放", "answer": true}
 ]}
```

- `answer` 是人工标注的真实情况（true/false）
- `name` 用短名（作报告列名和结果 ID）；**主观/程度类检查务必加 `criteria` 判定标准**，只进提示词不进报告，用来对齐模型与标注员的标准——这是主观项准确率的最大影响因素
- 模型对每个检查项返回 `verdict`（是/否）+ `reason`（一句话图片依据），依据存入结果并在报告错题样本中展示，便于区分"模型真错了"还是"标准有歧义"
- 报告自动按**检查项名**分列统计准确率（如"收银台干净整洁 71%"），一眼看出哪个场景是弱项
- 解析兼容多种变体（纯"是/否"、布尔、数组）；漏答/胡答按答错计并标注 `parse_fail`

### 单项提问模式

每行一个独立问题，适合不同图问不同事：

准备一个 JSONL（或同列 CSV），每行一题，图片路径相对数据集文件：

```json
{"id": "q001", "image": "images/001.jpg", "task": "classification",
 "question": "这张图里的产品是哪种型号？", "choices": ["A型", "B型", "C型"],
 "answer": "B型", "category": "型号识别"}
{"id": "q002", "image": "images/002.jpg", "task": "vqa",
 "question": "图中破损位置在左上、右上、左下还是右下？", "answer": "左下",
 "match": "judge", "category": "缺陷定位"}
{"id": "q003", "image": "images/003.jpg", "task": "vqa",
 "question": "图中共有几个安全隐患点？", "answer": "3", "match": "numeric", "category": "计数"}
```

字段说明：

| 字段 | 必填 | 说明 |
|---|---|---|
| `image` | ✅ | 图片路径；超过 4MB 或最长边超 1568px 会自动压缩 |
| `task` |  | `classification`（需 `choices`）/ `checklist`（需 `checks`）/ `vqa`，默认 `vqa` |
| `question` | 分类/VQA 必填 | 提问文本 |
| `answer` | 分类/VQA 必填 | 标准答案。分类题填选项文本或对应字母 |
| `choices` |  | 分类题选项列表 |
| `checks` |  | checklist 检查项列表 `[{name, answer, criteria?}]`，answer 为 true/false，criteria 为可选判定标准 |
| `match` |  | VQA 判分策略，默认有 judge 用 judge、否则 contains |
| `category` |  | 业务分类，报告中分目统计（checklist 用检查项 name，无需填） |
| `tolerance` |  | numeric 允许误差，默认 0 |

**VQA 判分策略怎么选**

- `numeric`：答案是数字（计数、读数）用这个，支持中文数字（"三个"）
- `exact`：答案唯一且无歧义（编号、是/否），忽略大小写和标点
- `contains`：标准答案需出现在回答里（宽松，适合关键词式答案）
- `judge`：LLM 裁判语义判分（推荐用于开放回答）。裁判只收到问题/标准答案/模型回答的**文本**，不会看图，所以任何纯文本模型都能当裁判。模型输出很长或语义开放时选这个最稳。

## 添加模型

编辑 `vlmeter/config/models.yaml`，加一段即可（把 `enabled` 改为 `true`）：

```yaml
  glm-4v:
    enabled: true
    provider: openai                 # OpenAI 兼容接口都用这个 provider
    model: glm-4v-plus
    base_url: https://open.bigmodel.cn/api/paas/v4/
    api_key_env: ZHIPUAI_API_KEY     # 从 .env / 环境变量读
    concurrency: 4                   # 并发请求数
```

本地模型两种方式：

```bash
# 方式一：vLLM（推荐，OpenAI 兼容）
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000
# 方式二：Ollama
ollama pull llava
```

对应 yaml 里已预置 `qwen2.5-vl-7b`（vLLM）和 `llava`（Ollama）条目，改 `enabled: true` 即可。

> 注意：若本机设置了 `OPENAI_BASE_URL` 代理（如本环境指向 DeepSeek），官方 OpenAI 条目必须写死 `base_url: https://api.openai.com/v1/` 并用独立密钥变量，避免把密钥发错服务端。

## 运行与报告

```bash
python3 -m vlmeter run --models a,b,c --data data/my.jsonl \
    --judge deepseek-v4-pro \         # 裁判模型（可选，纯文本模型即可）
    --limit 20 \                      # 先跑前 20 题试水
    --fresh                           # 忽略缓存全部重跑
python3 -m vlmeter report --data data/my.jsonl          # 随时重新出报告
```

- 中断/限流失败不用怕：再执行同一条命令，只补失败和缺失的题
- 结果文件按数据集内容指纹隔离，换数据自动开新文件，不会串
- 推理型模型（如 DeepSeek）思考耗尽 `max_tokens` 会返回空答案，运行器会自动重试换一次采样；仍频繁出现可在该模型配置里调大 `max_tokens`

## 准确率与成本怎么看

两个核心指标都直接落在报告里，配合做选型决策：

**准确率**（多层下钻，避免被单一数字误导）：
- 总览表：总体 / 分类 / 检查项 / VQA 分项准确率
- 分目表：**每个检查项一列**（如"收银台干净整洁 71%"），定位弱项场景
- 错误样本：标准答案 vs 模型输出 + 模型判定依据（reason），区分"真错"和"标准歧义"

**成本**（按实际用量核算，不是拍脑袋估）：
- 在 `vlmeter/config/models.yaml` 给模型配 `price_input` / `price_output`（元/百万 tokens，⚠️ 示例价需按官网核对）
- 每次调用的输入/输出 tokens 从 API 响应实时记录；checklist 一图多检查项按**一次调用**计费（自动去重）
- 报告"成本与性能"表：本次总成本、单次调用成本、**每 1000 次调用成本**（生产量级外推）、平均延迟，与准确率并排
- 本地模型（vLLM/Ollama）配 `price: 0`，API 成本计 0（只有 GPU 电费，不在本工具口径内）
- judge 裁判调用费用另计（评测基础设施，不是被测模型成本）

**典型权衡读法**：模型 A 准确率 98% 每千次 ¥30，模型 B 准确率 93% 每千次 ¥3——若业务允许 93%（如人工复核兜底），B 省十倍；若错误代价高（如合规检查），A 更划算。报告给了数字，决策是业务的事。

**统计口径**

- **准确率** = 判对的判定点 / 已判定判定点（checklist 一图展开为多个判定点；请求失败和未判定的不计入分母，别把断线当"答错"）
- **解析失败**：模型没按选项字母格式回答且无法映射回选项，按答错计（配置 judge 后裁判会把自由表述映射回选项，大幅减少这类误伤）
- **错误样本**：逐题列出标准答案 vs 模型输出，建议先看这个再下结论

## 项目结构

```
vlmeter/                 评测包
  models/              各家模型适配器（openai 兼容 / anthropic / gemini / ollama / mock）
  dataset.py           数据集加载与校验
  prompting.py         分类/VQA 提示模板
  scoring.py           选项解析与 exact/contains/numeric 判分
  judge.py             LLM 裁判（VQA 语义判分 + 分类自由回答映射）
  runner.py            异步并发、重试、断点续跑
  report.py            聚合报告
vlmeter/config/models.yaml     模型注册表
scripts/make_sample_data.py  合成示例数据
tests/                 判分单元测试 + 适配器冒烟测试
```

## 测试

```bash
python3 tests/test_scoring.py      # 判分逻辑
python3 tests/test_adapters.py     # 适配器请求格式（本地 mock 服务，不需要密钥）
```

## 发布新版本到 PyPI

一次性准备（只需做一次）：

1. 注册 [PyPI](https://pypi.org) 账号，在 **Account Settings → API tokens** 创建 token（account scope，项目名占住后可换项目 scope）
2. 把 token 写入 `~/.pypirc`：

   ```ini
   [distutils]
   index-servers =
       pypi

   [pypi]
   username = __token__
   password = pypi-你的token
   ```

每次发版四步（以 0.2.3 为例）：

```bash
# 1. 升版本号（两处都要改）
sed -i 's/__version__ = "0.2.2"/__version__ = "0.2.3"/' vlmeter/__init__.py
sed -i 's/version = "0.2.2"/version = "0.2.3"/' pyproject.toml

# 2. 回归 + 构建并校验（先清掉旧 dist，避免重复上传旧版本）
python3 tests/test_scoring.py && rm -rf dist && python3 -m build && python3 -m twine check dist/*

# 3. 上传（上传后 PyPI 索引有约半分钟延迟属正常）
python3 -m twine upload dist/*

# 4. 提交并打 tag 推送
git add vlmeter/__init__.py pyproject.toml && git commit -m "chore: 发布 v0.2.3" && git push
git tag v0.2.3 && git push origin v0.2.3
```

说明：

- **同一版本号不能重复上传**：PyPI 不允许覆盖已发布的文件，改了代码必须升版本号
- 构建隔离失败（部分受限网络环境）时用 `python3 -m build --no-isolation`，走系统已装的 setuptools
- 仓库里的 `.github/workflows/publish.yml` 是备用的 CI 发布通道（手动触发），配置 PyPI Trusted Publishing 后可切换；日常以本地 twine 为准
- 本地 twine 走 `~/.pypirc` 里的 token，`git push` 不会把任何密钥带上去

## 已知边界

- Ollama 适配器请求格式经过本地 mock 服务验证，但未在真实 Ollama 实例上跑过（本机无 GPU）；vLLM 路径与 DeepSeek 同走 OpenAI 兼容适配器，已真实验证
- 裁判模型自身有判错概率；关键结论建议抽查报告中的错误样本
- 数据集指纹基于 JSONL/CSV 文件内容，单独替换同名图片不会触发重跑（用 `--fresh` 强制）
