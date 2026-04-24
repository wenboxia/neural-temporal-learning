# neural_1 项目 — 轻量工作流配置蓝图

**日期**：2026-04-24
**目的**：给 neural_1 配一个务实的 Claude Code 工作流，让日常开发少点弹窗，关键任务能用子 agent 聚焦，不引入不必要的复杂度。本计划只做设计，不在当前窗口执行；带去新 Claude Code 窗口落地。

---

## Context

用户希望在 neural_1（PFC 启发的 TabPFN 三时间尺度系统）的重启阶段：

1. 日常开发**少点弹窗**，不打断思路（主要目标）
2. 能定义几个项目级角色，让 lead session 用 Agent 工具 spawn 他们做聚焦任务
3. 不引入超出实际需要的功能（Agent Teams 当前用不到，bypass 权限风险收益不成比例）

### 决策拍板结果

| 决策点 | 结论 |
|---|---|
| **权限模式** | `default` + `permissions.allow` 白名单（80%+ 弹窗消失）+ 短 deny 列表（兜底危险操作） |
| **Agent Teams** | **不默认启用**。保留"知道怎么临时开"的知识；等到 Phase 4 消融 / 多角度 review / 竞争假设 debugging 时单次加环境变量 |
| **角色定义** | 3 个项目级 agent：`fullstack-engineer` / `tester` / `experimenter`；通过**内置 Agent 工具**在单 session 内 spawn；不绑定 Agent Teams |
| **Skill 过滤** | 项目内仅保留 `update-config` 和 `fewer-permission-prompts`，其它隐藏（不删除，不影响其它项目） |
| **paper-writer** | 不定义，等项目完成后再加 |

---

## 待创建文件清单

4 个文件，全部在 `/Users/wenbo/Desktop/neural_1/.claude/` 下（项目级，不影响全局）。

### 1. `.claude/settings.json`

```json
{
  "permissions": {
    "allow": [
      "Bash(python *)",
      "Bash(python3 *)",
      "Bash(pytest *)",
      "Bash(git status)",
      "Bash(git status *)",
      "Bash(git diff *)",
      "Bash(git log *)",
      "Bash(git show *)",
      "Bash(git add *)",
      "Bash(git commit *)",
      "Bash(git restore *)",
      "Bash(git reset *)",
      "Bash(ls *)",
      "Bash(cat *)",
      "Bash(head *)",
      "Bash(tail *)",
      "Bash(wc *)",
      "Bash(grep *)",
      "Bash(rg *)",
      "Bash(find *)",
      "Bash(mkdir *)",
      "Bash(cp *)",
      "Bash(mv *)",
      "Edit(//Users/wenbo/Desktop/neural_1/**)",
      "Write(//Users/wenbo/Desktop/neural_1/**)",
      "Read(//Users/wenbo/Desktop/neural_1/**)"
    ],
    "deny": [
      "Edit(//Users/wenbo/.claude/**)",
      "Write(//Users/wenbo/.claude/**)",
      "Edit(//Users/wenbo/.agents/**)",
      "Write(//Users/wenbo/.agents/**)",
      "Bash(sudo *)",
      "Bash(rm -rf /*)",
      "Bash(rm -rf ~/*)",
      "Bash(rm -rf /Users/wenbo/*)",
      "Bash(curl *)",
      "Bash(wget *)",
      "Bash(pip install *)",
      "Bash(pip3 install *)",
      "Bash(pip uninstall *)",
      "Bash(brew install *)",
      "Bash(brew uninstall *)",
      "Bash(npm install *)",
      "Bash(ssh *)",
      "Bash(scp *)",
      "Bash(git config --global *)",
      "Bash(git push *)",
      "Bash(git remote *)"
    ]
  }
}
```

**关键说明**：
- **`defaultMode` 没写** → 使用默认（非 bypass），弹窗正常工作
- `allow` 里列的操作**自动通过无弹窗**；剩下没列到的仍会弹
- `deny` 即使在白名单匹配时也会优先拦截（保险栓）
- 跨项目的 `Edit/Write` 路径没列进 allow → 默认需要弹窗确认 → 等于隐式限制到 neural_1
- `Bash(git push *)` / `pip install *` 等被 deny —— 你不上 GitHub、不希望随意装包
- **skill 过滤字段**：不同 Claude Code 版本字段名可能是 `disabledSkills` / `skills.disabled` 等；执行窗口里让 Claude 用 `update-config` skill 或 `claude-code-guide` 子 agent 查准再加。目标效果：此项目内只看到 `update-config` 和 `fewer-permission-prompts`

### 2. `.claude/agents/fullstack-engineer.md`

```markdown
---
name: fullstack-engineer
description: 实现 neural_1 项目中的 Python/ML 代码（torch/numpy/sklearn），端到端完成模块开发：写类、写函数、接 pipeline、改 script 参数。收到具体规格时不要讨论架构，直接实现。
tools: Read, Write, Edit, Bash, Grep, Glob
model: sonnet
---

你是 neural_1 项目的全栈工程师。项目是基于 TabPFN 的多时间尺度时序学习系统，详见项目根目录的 CLAUDE.md 和 implementation_plan_v2.md。

## 职责范围
- 实现 `src/` 下的新模块（当前主要是 Phase 3：GatedEnsemble / fast_to_inter / MultiTimescaleModel）
- 修改 / 扩展已有模块
- 写 / 更新 `scripts/` 下的入口脚本（如 `run_phase3.py`）
- 修 bug、重构

## 职责边界（不做这些）
- **不写测试** —— 测试交给 `tester` 角色
- **不跑生产实验** —— 交给 `experimenter`
- **不做架构决策** —— 规格不明时向主 session 提问，不自作主张改计划
- **不 commit** —— 由主 session 或用户决定提交时机

## 风格约定
- 代码风格对齐 `src/models/fast_corrector.py` 和 `src/models/slow_prior.py`：中文 docstring，assert 形式的参数校验，模块顶部写"使用流程"示例
- 新文件开头注释说明"此模块属于 Phase X 的什么部分"
- **TabPFN 权重绝不微调**（V2 计划硬约束）
- CPU only，context_size ≤ 3000

## 交付形式
完成任务后在最终回复里列：
1. 写/改了哪些文件
2. 每个文件的核心类/函数名
3. 下一步建议（比如"等 tester 写完测试后，可以集成到 run_phase3.py"）
不要把完整代码块再贴回消息里 —— 调用方会直接看 diff。
```

### 3. `.claude/agents/tester.md`

```markdown
---
name: tester
description: 为 neural_1 项目写单元测试和集成验证。针对已实现的模块写 pytest 测试、跑现有测试、做 sanity check、catch 数值/形状/边界问题。
tools: Read, Write, Edit, Bash, Grep, Glob
model: sonnet
---

你是 neural_1 项目的测试工程师。现有测试在 `tests/` 下，参考 `tests/test_fast_corrector.py` 的风格。

## 职责范围
- 为新模块写单测（shape / dtype / 边界 / 数值 invariants）
- 跑 `pytest tests/` 并分析失败
- 为 Phase 3 新模块加集成测试（例如 GatedEnsemble 输出权重和为 1、MultiTimescaleModel 的 step 函数能正常跑 100 步）
- 对 ML 逻辑做 sanity check（例如 FastCorrector 的 correction 落在 [-1, 1]）

## 职责边界（不做这些）
- **不改实现代码** —— 发现 bug 时写测试暴露问题并向主 session 报告；让 `fullstack-engineer` 去修
- **不跑生产实验** —— 只跑 pytest；完整 `run_phase3.py` 交给 `experimenter`

## 风格约定
- 测试类名 `TestXxxBehavior`，方法名用完整英文描述（如 `test_knn_exact_match`）
- ML 模块测试用固定随机种子保证可重复
- 数值容差 `abs(a - b) < 1e-6`（浮点）或 `< 0.05`（统计行为）
- 不依赖 TabPFN 的测试（TabPFN 加载慢）优先，必须依赖时加 `@pytest.mark.slow`

## 交付形式
列出：
1. 新增测试文件 + 测试数量
2. pytest 结果（pass/fail 数字）
3. 发现的 bug 清单（如有）
```

### 4. `.claude/agents/experimenter.md`

```markdown
---
name: experimenter
description: 为 neural_1 项目跑实验脚本、分析结果、画图、写结果段落。不写代码实现、不写 src/ 下的模块。擅长解读准确率曲线、门控权重轨迹、漂移前后对比。
tools: Read, Bash, Write, Edit, Grep, Glob
model: sonnet
---

你是 neural_1 项目的实验分析员。

## 职责范围
- 跑 `scripts/run_phase*.py` 并记录结果
- 解读 `results/*.png` 和 `.npz` 输出
- 写 `progress_report.md` 的实验结果段落
- 画新图（如 Phase 3 的 gate 权重轨迹图）
- 做超参消融（手动跑不同 `--buffer_size` / `--fixed_ratio` / `--context_size`）
- 实验不符合预期时，**只做初步诊断 + 提出假设**；不改 src/ 代码，把诊断交给主 session

## 职责边界（不做这些）
- **不改 `src/` 下的实现代码** —— 发现实现问题交 `fullstack-engineer`
- **不改测试** —— 交 `tester`
- **不修改未 commit 的他人改动** —— 冲突时报告

## 关键约束（硬编码到脑子里）
- Prequential 协议：先预测 → 再观测 → 再更新
- CPU only，长实验慢（5000 步 n_estimators=4 约 30-40 min）；先 `--max_eval_steps 100` 快速验证，再全量跑
- 图保存到 `results/`；`.png` git 跟踪，`.npz` 被 gitignore 屏蔽

## 交付形式
1. 跑了什么命令（完整命令行）
2. 核心数值（总体准确率、漂移后准确率、适应速度）
3. 图的保存路径
4. 2-3 句结论："是否符合预期" + "下一步建议"

不要把 log 贴全文到消息里；关键数值拎出来就行。
```

---

## 执行顺序（新窗口里按这个顺序做）

**Step 1 — 环境 & 目录**
```bash
cd /Users/wenbo/Desktop/neural_1
mkdir -p .claude/agents
ls -la .claude/
```

**Step 2 — 创建 4 个文件**（本计划里的内容原样贴）
```
.claude/settings.json
.claude/agents/fullstack-engineer.md
.claude/agents/tester.md
.claude/agents/experimenter.md
```

**Step 3 — 加 skill 过滤字段**

settings.json 中加入 skill 隐藏字段。由于 Claude Code 版本的具体字段名可能不同，在新窗口里执行：

```
/update-config
"我想在本项目的 .claude/settings.json 里只启用 update-config 和 fewer-permission-prompts 两个 skill，其它 skill 全部在本项目隐藏（不要删除，不影响其它项目）。加对应字段。"
```

或者让 Claude 调用 `claude-code-guide` 子 agent 查准确写法再加。

**Step 4 — 重启 Claude 使 settings 生效**

退出当前进程，重新 `claude`。

---

## 验证（3 项 checklist）

| # | 测试 | 期望 |
|---|---|---|
| 1 | 让 Claude 跑 `python scripts/run_baselines.py --help` | **无弹窗**，直接执行 |
| 2 | 让 Claude 尝试 `Edit /Users/wenbo/.claude/foo.md` | **被 deny 拒绝** |
| 3 | 问 Claude："现在能看到哪些 skill？" | **只看到 `update-config` 和 `fewer-permission-prompts`** |

3 项都通过 → 配置生效。

---

## 临时场景：什么时候打破这套基线？

以下两种情况**不改 settings.json**，用单次命令行参数：

### 情况 A：临时想通宵无人值守跑实验
```bash
claude --dangerously-skip-permissions
```
这次 session 期间全放行（deny 列表仍生效）。退出后下次启动恢复正常。

### 情况 B：真需要 Agent Teams 并行（Phase 4 消融、多角度 review、竞争假设 debugging）
```bash
CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1 claude
```
临时启用；退出后下次启动不带 Agent Teams。具体使用指南见官方文档（https://code.claude.com/docs/en/agent-teams）。

触发 Agent Teams 的 4 类场景（日常都不用、遇到了再开）：
1. 并行实验批跑（3 数据集 × N 超参）
2. 多角度 code review（correctness / ML 逻辑 / 测试）
3. 假设调试（competing hypotheses 互相反驳）
4. 论文阶段 section 并行起草（paper-writer 角色后续加）

---

## 日常怎么用这三个 agent（单 session 里）

不启用 Agent Teams 的情况下，在主 session 里直接说："帮我用 fullstack-engineer 实现 `src/models/gated_ensemble.py`"，Claude 会通过**内置 Agent 工具** spawn 这个角色，它会加载 `.claude/agents/fullstack-engineer.md` 作为 system prompt，在独立 context 里执行，完成后把结果返回给主 session。

这跟 Agent Teams 的核心区别：
- 子 agent 只向主 session 汇报，不能互相直接通信
- 子 agent 完成即结束，不会持续存在
- **对 Phase 3 这种顺序开发完全够用**

---

## 有意未包含

- **Hooks**（`TaskCompleted` / `TeammateIdle`）—— Agent Teams 相关，当前不启用
- **Routines / scheduled agents** —— 不需要定时触发
- **paper-writer 角色** —— 项目完成后再加
- **MCP servers** —— 本项目离线为主
- **自定义 slash commands**（如 `/run-phase3`）—— 工作流稳定后再提炼

---

## 风险提示

1. **allow 白名单有盲点**：如果 Claude 使用了没列进 allow 的命令（比如 `uv run python` 或新发明的组合），仍会弹窗 —— 这是**特性不是 bug**，提醒你看一眼。如果发现某条经常弹且明确安全，让 Claude 用 `fewer-permission-prompts` skill 扫历史自动补 allow。

2. **deny 仍不是铁壁**：glob 可能漏 edge case。但在 `default` 模式下，未 allow 的操作仍会弹窗二次确认 —— 即使 deny 漏了，你仍有机会拦住。瘦身版方案的**多层防御是 allow 精挑 + deny 兜底 + default 模式的人工确认**三道。

3. **skill 字段名依赖版本**：Step 3 真正执行时让窗口里的 Claude 用 `update-config` skill 或查文档确认。

4. **git 兜底**：每完成一个完整任务让 Claude commit 一次。`git reflog` 可恢复意外改动。

5. **路径 glob 必须用 `//` 前缀**：路径类 allow/deny（`Edit(...)` / `Write(...)` / `Read(...)`）必须用 `//` 前缀（Claude Code 内部路径规范化），单 `/` 无效；该格式可能因 Claude Code 版本变更，升级后需复验。

---

## 当前项目背景（给新窗口 lead 的速览）

- Git 有 6 个本地 commit（`23b7ae3` 初始 + 5 个 Phase 2.5 commit），无 GitHub remote
- Phase 1 ✅ / Phase 2 ✅ / Phase 2.5 ✅ / **Phase 3 待启动**
- Phase 3 的 4 个待创建代码文件：
  - `src/models/gated_ensemble.py`
  - `src/consolidation/fast_to_inter.py`
  - `src/models/multi_timescale.py`
  - `scripts/run_phase3.py`
- 代码模板：`implementation_plan_v2.md` 第 231-311 行
- 验收标准：`implementation_plan_v2.md` 第 468-472 行

---

## End-to-end 验证

新窗口走完 Step 1-4 + 通过 Step 5 的 3 项 checklist = 配置落地成功。

之后建议第一个实际任务：**让 `fullstack-engineer` 实现 Phase 3A GatedEnsemble，然后 `tester` 写单测**。范围小、依赖少，最容易验证这套工作流有没有副作用。
