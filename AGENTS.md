# AGENTS.md — vlmeter 工作区指南

## 项目

vlmeter：多模态大模型准确率 × 成本计量流水线。自有图片+标注数据评测视觉语言模型（checklist 场景检查 / classification / vqa），产出 Markdown + 自包含 HTML 对比报告。已发布 [PyPI](https://pypi.org/project/vlmeter/) 与 GitHub（flybear16/vlmeter，公有）。

## 目录与命令

```
vlmeter/                  包（pip install vlmeter 即 CLI）
  models/                 适配器：openai 兼容(主干)/anthropic/gemini/ollama/mock
  dataset.py prompting.py scoring.py judge.py
  runner.py               异步并发/重试/断点续跑
  report.py report_html.py cli.py
  config/models.yaml      模型注册表——随包分发的数据文件，不是仓库根配置
scripts/make_sample_data.py  合成示例数据（data/ 可再生成的产物）
tests/                    判分单测 + 适配器契约测试（本地 mock HTTP 服务）
docs/PRD.md               产品需求文档
```

```bash
python3 tests/test_scoring.py                    # 改判分/scoring 后的第一道回归
python3 tests/test_adapters.py                   # 改适配器请求格式后必跑
python3 -m vlmeter run --models mock --data data/sample.jsonl   # 零成本端到端
```

无 lint/typecheck 配置；Python ≥3.10，风格：类型注解 + dataclass。

## 架构规则

- **加模型只改 yaml 不改代码**（provider/model/base_url/api_key_env/单价）；新 provider 才写适配器，需有 `provider_name` 类属性并在 `vlmeter/models/__init__.py` 注册
- **判定点粒度**：checklist 一图展开为 `图ID::检查项名` 多条记录；缓存、续跑、统计都以判定点为单位。续跑时先重写结果文件清掉失败/未判定行，防重复统计
- **打分分层**：确定式优先（选项匹配/exact/contains/numeric），LLM judge 只兜底（VQA 语义判分 + 分类自由回答映射）；judge 只收文本不看图
- **成本按调用去重**：checklist 多记录共享一次调用的 tokens，报告侧按 `id::` 前缀去重计费
- 版本号在**两处**：`vlmeter/__init__.py` + `pyproject.toml`，必须同步改；PyPI 同版本号不可重传

## 本机环境坑（重要）

- `OPENAI_BASE_URL` 指向 DeepSeek 且 key 可用：`deepseek-flash`（视觉）、`deepseek-v4-pro`（纯文本，当 judge）开箱即用
- 本会话环境带 SOCKS 代理变量：新 venv 的 pip 会吊死（缺 PySocks）；**用系统 pip3**。httpx 的 socksio 已装
- `python3 -m build` 隔离构建会失败（PEP 668），用 `--no-isolation`
- 官方 OpenAI 条目必须写死 base_url + 独立密钥变量，防止把 DeepSeek key 发错服务端

## 发布（详见 README「Publishing a release to PyPI」）

升版本 → 测试 + `rm -rf dist && python3 -m build --no-isolation && twine check` → `twine upload`（token 在 `~/.pypirc`）→ commit + tag `vX.Y.Z` 推送。GitHub publish workflow 是手动触发的备用通道（trusted publishing 未配好前 tag 触发必失败）。

## 约定

- 面向用户的文案（报告、README、报错）用中文；代码注释/标识符用英文
- README 双语：`README.md`（英文主文档）+ `README.zh-CN.md`，内容保持同步，顶部互链
- 改任务语义/判分口径前先读 `docs/PRD.md`；`tests/test_scoring.py` 的对抗样例（枚举选项、加粗字母）来自真实模型输出，不可删
- `data/`、`results/`、`dist/` 均为产物，已 gitignore，不入库
