# ADR-0002: chunking 策略选型

## 上下文
RAG 检索质量强依赖 chunk 粒度, 直接用默认值无决策依据, 需明确切分策略并支持变体对比。

## 选项
- **fixed**: 按段落定长切, 最朴素基线
- **recursive**: 递归字符切分(多级分隔符), 平衡粒度与语义
- **markdown**: 按标题结构切, 适合结构化文档

## 决策
默认用 `recursive(chunk_size=800, overlap=120)`。

## 理由
- 论文 PDF 经 pymupdf 解析后无明确 markdown 结构, markdown 策略优势不明显
- recursive 的多级分隔符(`\n\n` -> `\n` -> `. ` -> 空格)尽量保持语义完整
- `chunk_size=800` 兼顾上下文充分与检索精度; `overlap=120` 缓解切分断句
- `config.yaml` 的 `variants` 保留 fixed/markdown 用于评估对比

## 后果
- **正面**: 切分可解释、可对比
- **负面**: 论文表格/公式可能被切碎(pymupdf 文本提取局限)
- **后续**: 可对论文结构化段落(摘要/引言/方法)做语义切分进一步优化
