---
name: arxiv-research
description: 自动获取 arxiv 论文、深度阅读并生成中文研究报告（包含伪代码实现）
argument-hint: <研究主题，如 "LLM Agent" 或 "LLM 推理优化">
level: 3
---

# Arxiv 深度论文研究技能

自动从 arxiv 获取论文、下载 PDF、深度阅读核心方法，并生成包含伪代码实现的详细报告。

## 功能

1. **论文获取**：通过 arxiv API 获取指定主题的最新论文
2. **PDF 下载**：自动下载论文 PDF 全文
3. **深度阅读**：解析 PDF 提取核心方法、贡献和实验结果
4. **伪代码生成**：根据论文描述生成关键方法的伪代码实现
5. **Obsidian 友好**：生成带 front-matter 和标签的中文报告

## 使用方式

### Claude Code 中使用

```
/arxiv-research LLM Agent 系统最新进展
/arxiv-research 深度学习模型优化
```

### 命令行使用

```bash
# 获取论文并生成深度报告
python scripts/arxiv-deep-research.py --topic "LLM Agent" --days 7

# 指定输出目录
python scripts/arxiv-deep-research.py --topic "transformer" --output ./reports/
```

## 研究流程

```
1. 搜索 arxiv API
       ↓
2. 获取论文列表（摘要）
       ↓
3. 下载 Top N 论文 PDF
       ↓
4. 深度阅读 PDF
       ↓
5. 提取：核心问题、贡献、方法、实验结果
       ↓
6. 生成伪代码实现
       ↓
7. 输出中文报告（Markdown）
```

## 报告格式

每个报告包含：
- **Front-matter**：Obsidian 兼容 tags、日期、类型
- **论文摘要**：核心问题和方法概述
- **核心贡献**：主要创新点
- **方法详解**：数学形式化 + 架构图描述
- **伪代码实现**：关键方法的代码片段
- **实验结果**：关键数据和对比
- **实践洞察**：可落地的建议

## 研究方向

默认覆盖四个方向：
1. **评估基准**：benchmark、evaluation、performance
2. **推理优化**：quantization、distillation、inference
3. **Agent/RAG**：agent、RAG、retrieval、tool
4. **架构改进**：architecture、attention、transformer

## 目录结构

```
.
├── scripts/
│   ├── arxiv-llm-weekly-report.py    # 周报脚本
│   └── arxiv-deep-research.py       # 深度研究脚本
├── papers/                          # 下载的 PDF
├── deep-research-report-{date}.md   # 深度报告
└── arxiv-llm-research-{date}.md     # 周报
```

## 注意事项

- PDF 解析需要 `pymupdf` 库：`pip install pymupdf`
- API 请求有频率限制，每次查询间隔建议 3 秒
- 建议每次深度研究选择 2-3 篇最有价值的论文
- 伪代码基于论文描述，实际实现可能需要调整
