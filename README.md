# 大模型第三课 · 作业 3（A + B 两个方向都做）

> 课件：《知识的深加工》p163「作业 3：两个方向，任选一个」
> 这里 **A、B 都做了**：A 建了一个白酒 11 家的财报问答知识库，B 用茅台财报训了一个「董秘」LoRA。
> 语料来源：**巨潮资讯网**（交易所指定披露平台）→ `https://www.cninfo.com.cn`

## 当前状态

| # | 交付物 | 状态 |
|---|---|---|
| A | 代码仓库（**本仓库，已公开**） | ✅ `github.com/bh2009wan-crypto/cn-filings-rag` |
| A | 11 家 × 2026 半年报的知识库（向量 + BM25） | ✅ 4,462 块，向量 + BM25 双索引 |
| A | 问答页面（能提问、答案带出处） | ✅ `scripts/08_app.py`（答案带角标 + 证据卡片 + 数字核对） |
| A | 页面截图 | ✅ `outputs/A_页面截图.png`、`outputs/A_页面截图_回答.png` |
| A | 10 道题逐题记录（召回块 / 对错 / 错在哪） | ✅ `outputs/eval/results_a.csv`（含人工判分列） |
| A | 一页结论 | ✅ `docs/结论A_一页.md`（单公司 8/8，跨公司全景 0/2） |
| B | 造数据脚本 + 抽检记录 | ✅ 质检 312 → 人工抽检弃用 14 → **298 条**；训练 254 / 测试 44；换问法 246 与 44；答案带出处引用。抽检 40 条**已终审完毕**（26 留 / 14 删），见 `docs/抽检记录.md` + `data/sft/review_40.csv` |
| B | 训练笔记本 | ✅ `notebooks/maotai_dongmi_lora.ipynb`（Colab 免费 T4；仓库已公开，4 个数据文件自动拉取，只有 `rag_context.json` 需手动上传）<br>**交付清单见 `docs/交作业清单.md`** |
| B | 四格对照表 + 一页结论 | ✅ **本机 M2 跑完**：`outputs/B_对照表.md`、`docs/结论B_一页.md`（基座 Qwen3-0.6B；格式学会、事实学不会；Colab 版笔记本仍可用） |
| B | 本机训练脚本（Colab 被网络卡死后的替代路） | ✅ `scripts/23_local_sft_mps.py`（分阶段跑，bf16，51 步 / 8.8 分钟） |

> 状态如实写：**没跑完的不打勾**。

## 交付物 ↔ 文件对照（交作业时按这张表找）

| 课件要求的交付物 | 文件 |
|---|---|
| A · 代码仓库 | 本仓库 |
| A · 页面截图 | `outputs/A_页面截图.png`（首页）、`outputs/A_页面截图_回答.png`（带证据卡片的回答） |
| A · 一页结论 | `docs/结论A_一页.md` |
| A · 10 题逐题记录 | `outputs/eval/results_a.csv`（+ 可复核的判分脚本 `scripts/19_verdict_a.py`） |
| A · 冻结的题与参考要点 | `eval/questions_a.json`、`eval/questions_a.sha256`、`eval/key_facts.json` |
| B · 数据生成脚本 | `scripts/10_gen_qa.py`、`scripts/11_qc_qa.py` |
| B · 人工抽检记录 | `docs/抽检记录.md` + `data/sft/review_40.csv`（**待你填判定列**） |
| B · 训练笔记本 | `notebooks/maotai_dongmi_lora.ipynb` |
| B · 对照表 | `outputs/B_对照表.md`（等 Colab 结果） |
| B · 一页结论 | `docs/结论B_一页.md` |
| 附 · 踩坑复盘（不在要求里，但对同类项目有用） | `docs/踩坑复盘.md`（14 条） |
| 附 · 你要做的两步（Colab 可选 + 人工抽检） | `docs/你要做的两步.md` |

## 课件要求 ↔ 实现 对照

### A · 下载财报，做一个问答知识库

| 课件要求 | 实现 | 脚本 |
|---|---|---|
| ① 一个行业或一组公司（10 家以上），从交易所网站下载年报/半年报全文 | **白酒 11 家 × 2026 年半年报**，从巨潮下载 PDF | `01_resolve_orgids.py`、`02_fetch_reports.py` |
| ② 提取文字，表格按行列还原；切块时带上公司、章节、页码 | pdfplumber 抽正文 + 表格还原（含跨页表合并）；切块 800 字 / 重合 120，四标签 | `03_parse_pdf.py`、`04_chunk.py` |
| ③ 建向量 ＋ BM25 索引，做一个能提问、答案带出处的问答页面 | Qwen3-Embedding-0.6B + 自实现 BM25 → RRF 融合 → Flask 页面（答案带 `[n]` 角标 + 证据卡片） | `05_build_bm25.py`、`06_embed.py`、`ask_lib.py`、`08_app.py` |
| ④ 出 10 道题（至少 2 道跨公司全景题），逐题记录 | 10 道（含 2 道跨公司全景）+ 1 道附加拒答题；**先冻题再跑**（哈希入库） | `20_build_eval_set.py`、`21_run_eval_a.py` |
| 交付：代码仓库 ＋ 页面截图 ＋ 一页结论 | 本仓库 + `outputs/A_页面截图*.png` + `docs/结论A_一页.md` | `09_screenshot.py` |

### B · 用财报训练一个「董秘」

| 课件要求 | 实现 | 脚本 |
|---|---|---|
| ① 一家公司，近几年的年报 + 半年报 | 贵州茅台 2021–2025 年报 + 2021H1–2026H1 半年报 = **11 份**（2026 年报尚未披露，约 2027-04 才出） | `02_fetch_reports.py --set B` |
| ② 大模型从原文生成问答，注明出自哪份报告哪一页；人工抽检删编造 | 提问/作答/质检三角色同一个便宜模型（deepseek-flash）；每条**强制逐字抄"依据原文"**好做校验；抽检 40 条（固定种子、分层） | `10_gen_qa.py`、`11_qc_qa.py`、`12_sample_review.py` |
| ③ Colab 上用 LoRA 微调一个小模型（Qwen3.5-2B） | `notebooks/maotai_dongmi_lora.ipynb`（免费 T4、4bit、只训 LoRA；含模型不可用时的自动降级） | 同上 |
| ④ 四格对照（训练过/没过 × 原问法/换问法）；再和「A 的知识库 ＋ 小模型」比 | 笔记本里一次跑完四格 + RAG 对照；划分**按块**防泄漏 | `13_split.py`、`14_make_paraphrase.py`、`16_dump_rag_for_b.py`、`22_run_eval_b.py` |
| 交付：脚本 ＋ 抽检记录 ＋ 笔记本 ＋ 对照表 ＋ 一页结论 | 全套见上 | — |

## 目录结构

```
~/work/homework3/
├── config.py                  公司清单(代码/简称)、路径、切块 800/120、检索参数
├── config/companies.json      11 家的 orgId（01 脚本现场解析，不能猜——见下"踩坑"）
├── utils/                     envload(.env 唯一加载器) / http(退避重试) / llm(DeepSeek)
├── scripts/
│   ├── 00_env_check.py        开工前体检（解释器/MPS/巨潮/hf-mirror/.env/模型 ping）
│   ├── 01_resolve_orgids.py   解析 11 家 orgId（错一个就静默返回 0 条公告）
│   ├── 02_fetch_reports.py    巨潮下载：查询→筛选→去重(同公告)→下载→manifest
│   ├── 03_parse_pdf.py        页眉页脚/页码偏移/章节双路/表格还原+跨页合并+表题回溯
│   ├── 04_chunk.py            切块 800/120 + 四标签（标签拼进 index_text）
│   ├── 05_build_bm25.py       jieba + 自实现 BM25（不装 rank_bm25）
│   ├── 06_embed.py            Qwen3-Embedding-0.6B（MPS，last-token pooling，断点续跑）
│   ├── ask_lib.py             检索（BM25 ∪ 向量 → RRF；跨公司 fanout）+ 作答 + 后置校验
│   ├── 07_ask.py              命令行问一句
│   ├── 08_app.py              Flask 问答页面（含 /api/ask）
│   ├── 09_screenshot.py       selenium 截图（交付物）
│   ├── 10_gen_qa.py           造数据（一块 3 问，强制逐字"依据原文"）
│   ├── 11_qc_qa.py            质检：确定性校验 → 模型判"有据 0/1/2 · 自然 0/1"
│   ├── 12_sample_review.py    人工抽检样本（固定种子分层 40 条）
│   ├── 13_split.py            按"块"划分训练/测试（四格防泄漏的命门）
│   ├── 14_make_paraphrase.py  换问法生成（3-gram Jaccard < 0.5 才算换了）
│   ├── 15_make_colab_notebook.py  生成 Colab 笔记本
│   ├── 16_dump_rag_for_b.py   给 B 的 RAG 对照准备检索上下文
│   ├── 20_build_eval_set.py   校验 10 道题 + 冻存哈希
│   ├── 21_run_eval_a.py       跑 10 题 → results_a.csv
│   └── 22_run_eval_b.py       Colab 结果 → B_对照表.md
├── eval/                      questions_a.json（+ .sha256 冻题哈希）/ key_facts.json（参考要点）
├── data/                      pdf(忽略) text(忽略) chunks(统计入库) sft(问答对入库)
├── models/                    Qwen3-Embedding-0.6B（1.14GB，不入库）
├── notebooks/                 maotai_dongmi_lora.ipynb
├── docs/                      结论A_一页.md / 结论B_一页.md / 抽检记录.md / 踩坑复盘.md
└── outputs/                   页面截图 / eval 结果 / 对照表（md 入库，其余忽略）
```

## 怎么跑（从零复现）

```bash
cd ~/work/homework3
CR=/Users/lionel/anaconda3/envs/crawler/bin/python     # 抓取/解析/切块
NL=/Users/lionel/anaconda3/envs/nlp/bin/python         # 向量/BM25/页面（有 torch）

$CR scripts/01_resolve_orgids.py                       # ① 解析 orgId
$CR scripts/02_fetch_reports.py --set all              # ② 下载 11 + 11 份 PDF
$CR scripts/03_parse_pdf.py --set all                  # ③ 解析（11 家 + 茅台历年）
$CR scripts/04_chunk.py --set all                      # ④ 切块
$NL scripts/05_build_bm25.py --set A                   # ⑤ BM25
$NL scripts/06_embed.py --set A --device mps           # ⑥ 向量（首次会下载模型）
$NL scripts/08_app.py --port 7860                      # ⑧ 页面 → 浏览器打开
$CR scripts/09_screenshot.py                           # ⑨ 截图
$CR scripts/20_build_eval_set.py && $NL scripts/21_run_eval_a.py   # ④ 评测
$CR scripts/10_gen_qa.py --per-doc 18 && $CR scripts/11_qc_qa.py   # B 造数据+质检
$CR scripts/12_sample_review.py && $CR scripts/13_split.py && $CR scripts/14_make_paraphrase.py
$NL scripts/06_embed.py --set B && $NL scripts/16_dump_rag_for_b.py # B 的 RAG 对照上下文
# 然后把 data/sft/*.jsonl 传上 Colab 跑 notebooks/maotai_dongmi_lora.ipynb
```

## 数据与语料

| 语料 | 规模 | 来源 |
|---|---|---|
| A：白酒 11 家 2026 年半年报 | 1,531 页 / 63.9 万字正文 / 2,048 张表 / **4,462 块** | 巨潮（上交所/深交所指定披露平台） |
| B：贵州茅台 2021–2026（11 份） | 1,292 页 / 48.8 万字正文 / 2,238 张表 / **4,405 块** | 同上 |

> 课件里那套是 2,307 份半年报、4.56 亿字、703,012 块——规模差三个数量级，但**流程完全一致**；
> 我们这个量级用 numpy 存向量就够（课件用 LanceDB），省掉一个依赖，见"与课件的差异"。

## 分工：哪一步是脚本，哪一步是模型

| 步骤 | 谁做 | 为什么 |
|---|---|---|
| 下载/解析/切块/索引/统计 | 脚本 | 确定性的事不给模型 |
| 挑哪些块进窗口（检索） | 脚本（BM25 ∪ 向量 → RRF） | 可复现、可审计 |
| 只依据给定块作答 | 模型（deepseek-flash） | 判断与措辞 |
| 答案里的每个数字能否回原文 | 脚本（`number_check`） | 把"数字纪律"变成可测指标 |
| 造数据的提问/作答/质检 | 模型（三个角色同一模型，靠提示词分工） | 对齐课件做法 |
| 抽检删编造、四格结论 | **人** | 课件明写"人工抽检" |

## 与课件的差异（都写在这里，不藏着）

1. **向量库**：课件用 LanceDB；我们 4,462 块用 numpy 点积（毫秒级），省一个依赖。
2. **切块参数**：课件的半年报库用 800 字 / 重合 120，我们照抄；**表格单独成块**，并把超大表按行分段、每段重复表头（课件是把整张表放一块）。
3. **embedding**：课件用 Qwen3-Embedding-0.6B（本机），我们同款；但**不走 sentence-transformers**，用 `AutoModel` + last-token pooling，少装一个库。
4. **跨公司全景题**：课件靠 Agentic 逐层下钻/图谱收敛；我们在检索层加了 **fanout**（对 11 家各召回 2 块）——因为全局 top-8 不可能覆盖 11 家（课件原话：Agentic 挑到 11 家、窗口只放得下 4 家）。
5. **提示词注入日期**：课件没提，我们**必须注入**"今天是 2026年9月24日"——实测模型内部日期是 2026-05-07，不注入它会一口咬定 2026 半年报"尚未披露"（假性拒答，见踩坑）。
6. **B 的 RAG 对照语料**：四格的第 5 格在**茅台 11 份报告**上检索（而不是只用 A 的 2026H1 索引），否则跨年份的测试题会因为"库里根本没有"而失真——对比要公平。

## 踩坑复盘（每条都真踩过，带证据）

1. **orgId 不能从股票代码推**。按 `gssh0+代码` / `gssz0+代码` 猜：11 家里 **4 家猜错**（洋河、今世缘、口子窖、迎驾贡酒），而错的表现是**静默返回 0 条公告**——看起来像"这家没发财报"。必须调 `topSearch/query` 拿真值（有的是数字 id，如 `9900008689`），并把"返回 0 条"当**错误**而不是常态。
2. **表格文字会在正文里重复一遍**。实测量化：茅台 p27 不剔除时 979 字、剔除表格 bbox 内字符后 28 字。不处理的话正文块里塞的是表格的散字，检索质量直接塌。→ 用 `page.filter()` 抠掉表格区域。
3. **页码偏移逐份不同**。茅台页脚是 `1/110`（偏移 0）、古井贡酒是 `~ 141 ~` 且末页是 PDF 第 142 页（偏移 +1）、五粮液/泸州老窖/洋河页脚就是一个裸数字。→ 逐份探测页脚形态、取**众数**偏移、一致性 < 90% 就不用（只留 pdf_page）。所以每条引用都同时存 `pdf_page` 与 `printed_page`。
4. **同一页会出现两个节首**。茅台 p25 同时有"第七节 债券相关情况"和"第八节 财务报告"；一开始按"每页记第一个"处理，结果 **86 页财务报告全被标成"第七节 债券相关情况"**。→ 改成按"页内行号"记录所有节首，切块时按行号分段，块继承所在段的章节。
5. **表题回溯会抓到"表内行"**。pdfplumber 有时把表的前一两行漏在 bbox 外，那些行就成了"表上方的文字"，朴素做法会把「结算备付金」「向中央银行借款」当表题（更糟的是接着用表题做跨页合并判断）。→ 表题只认带"表/情况/状况/明细"关键词的行；找不到宁可留空。
6. **表内标题 ≠ 只有一格有字的行**。「结算备付金」这类表内行也常常只有一格有字。→ 改成用**单元格几何**判断：只有当首行那一个单元格跨越表格宽度 70% 以上（真的是合并单元格）才当表内标题。
7. **跨页表必须合并**。茅台合并资产负债表横跨 p25–p28（105 行）；不合并就出现"表头在第 1 段、数字在第 3 段"，`所有者权益合计` 这种行会被割裂。→ 页面贴底 + 下页贴顶 + 列数一致 + 同一章节 + 后表不带自己的表头，才合并。
8. **模型会假性拒答**。实测：问"2026 上半年营收"，模型答"当前 2026年5月7日，半年报尚未披露"——它按**内部日期**判断。→ 提示词首行注入"今天是 2026年9月24日"并明令"不要用内部知识判断报告是否已发布"；评测里单列 E6 错误码统计这类错误。
9. **thinking 块会吃光 token**。`max_tokens=16` 时全部被 reasoning 吃光、正文为空。→ 一律传 `thinking={"type":"disabled"}`（实测输出 token 132 → 1），质检等需要"想一下"的环节才开。
10. **hf-mirror 下不动大模型**。直连测速 7.5 KB/s、单连接 150 KB/s（1.14GB 要 2 小时），且多次 `SSL: UNEXPECTED_EOF`；huggingface.co 走 Clash 代理拿 API 能用、拿 LFS 文件超时。→ 改 **ModelScope**（842 KB/s）并**8 路并行分片 + Range**，总量 1,191,586,416 字节合并后校验大小一致。
11. **同一份公告会被两个清单各下一份**（茅台 2026 半年报同时在 A 和 B 的清单里），导致同一份报告被切两次块、训练数据重复。→ 按 `adjunctUrl` 去重，别名条目 `alias_of` 指向主条目，解析/切块时跳过。
12. **半年报的报告期标签会被写错**。`period.replace("H1","")` 把 `2021H1` 变成了 `2021`，于是"2021 年报"和"2021 半年报"共用一个文本文件名、**互相覆盖**。→ 报告期一律保留 `H1` 后缀，重新解析。

## 常见报错表

| 报错 | 原因 | 处理 |
|---|---|---|
| 公告列表返回 0 条 | orgId 猜错了（或 column 传错） | 先跑 `01_resolve_orgids.py --force` |
| `❌ 下载到的不是 PDF` | 巨潮返回了 HTML 错误页 | 检查 adjunctUrl；脚本会重试 |
| `pdftoppm is not installed` | Read 工具渲染 PDF 需要 poppler | 本作业不用它：走 pdfplumber 抽文字层 |
| `Operation not permitted`（列目录） | macOS TCC 保护桌面/文稿/下载 | 项目放 `~/work`，别放桌面；按**完整路径**读写单个文件是通的 |
| `SSL: UNEXPECTED_EOF_WHILE_READING` | hf-mirror 拉大文件被打断 | 换 ModelScope + 并行分片（见踩坑 10） |
| 页面卡住不返回 | 首次请求要加载 1.14GB embedding 模型 | 等 10–20 秒；后续请求很快 |
| 模型答"尚未披露" | 提示词没注入日期 | 见踩坑 8：`SYSTEM` 里必须有日期与"不要用内部知识判断" |

## 安全与合规

- `.env` 600 权限、不入库；密钥不写进代码、不截图。
- 财报 PDF 与派生文本（`data/pdf`、`data/text`、`data/index`、`models`）不入库；**入库的只有**：清单（`config/*.json`）、切块统计、SFT 问答对（`data/sft/*.jsonl`，无财报全文）、代码与结论。
- 巨潮请求间隔 1.5 秒 + 退避重试，礼貌抓取。
