# Git 上手指南(组员版)

> 面向第一次协作的新组员。读这一篇就能开始干活:配置 → 拿代码 → 日常循环 →
> 出问题怎么救。提交信息规范、入库红线与仓库《代码与提交规范》一致。

---

## 0. 一次性配置(装好 Git 后只做一次)

```bash
git config --global user.name  "你的名字"        # 署名,会用在你所有提交上
git config --global user.email "你的邮箱"        # 建议用注册 GitHub 的邮箱
```

验证:`git config --global --list`

## 1. 拿到代码

```bash
# 第一次:克隆到本地
git clone https://github.com/aaf12329/EyeWheelchairProject.git
cd EyeWheelchairProject

# 以后每次开始干活前:拉取最新代码
git pull
```

## 2. 分支规则 ★(本仓库约定)

| 分支 | 谁用 | 规则 |
|---|---|---|
| `main` | 所有者维护 | **组员不要直接 push**,由所有者合并 |
| `huang` / `chen` | 各自的开发分支 | 每人只推自己的分支,干完活合并进 main |

```bash
# 切到自己名下的分支(第一次,本地会自动跟踪远程同名分支)
git switch huang          # 或 chen

# 看看当前在哪个分支、本地有哪些分支
git branch -a
```

> 原则:**一人一分支,只推自己分支**。main 坏了所有人都停摆,所以合并由所有者把关。

## 3. 日常循环(每天就是这一套)

```bash
git status                    # ① 看改了什么(红=未跟踪/删除,绿=待提交)
git add 文件名                 # ② 挑要提交的文件(add . = 全部,慎用)
git commit                    # ③ 提交(信息怎么写见第 4 节)
git pull                      # ④ 拉别人的更新(在 main 上时尤其重要)
git push                      # ⑤ 推到自己分支
```

建议**小步提交**:一个逻辑单元一次 commit,别攒三天一起提。

### 关于 push:只推"当前所在分支" ★

`git push` 不带参数**只推当前所在的分支**,而且只推已经 commit 的历史
(改了文件没 commit,push 完远程也不会有)。几种写法的区别:

| 命令 | 推什么 |
|---|---|
| `git push` | 只推**当前分支**(前提:这个分支已经用 `-u` 设过上游) |
| `git push -u origin huang` | 第一次推自己的分支;`-u` 设上游,之后光 `git push` 就行 |
| `git push origin main huang chen` | 一条命令指定推多个分支(所有者合并时用) |
| `git push --all origin` | 本地所有分支一次全推(慎用,先 `git branch -a` 确认都有什么) |

## 4. 提交信息怎么写 ★(中英双语,本仓库格式)

一行说清"做了什么",中文在前,`CH:`/`EN:` 各一行:

```text
CH:新增串口输出层与美化界面(默认模拟模式),README 同步更新
EN:Add serial output layer and restyled UI (simulated by default); README updated accordingly
```

标题 ≤ 72 字符;细节(怎么验证的、影响哪些文档)放正文空行之后。

## 5. 红线:这些东西永远不要 `git add`

- 密钥、API key、`.env` 等凭据文件
- `myvenv/` 等虚拟环境目录(已被 .gitignore 挡住,别硬塞)
- `__pycache__/` 等缓存与生成物
- 大体积原始数据/视频/模型(先在群里问,再决定放哪)

> 不确定要不要提交?看 `.gitignore`,或者直接问。

## 6. 出问题了怎么救

```bash
# 改乱了工作区,想丢弃某个文件的未提交修改(不可恢复,慎用)
git restore 文件名

# commit 完发现少加/多加了东西,想撤回最近一次提交(改动保留在工作区)
git reset --soft HEAD~1

# 看提交历史(图形化分支线)
git log --oneline --graph --all -15

# push 被拒(远程有新提交)→ 先拉再推
git pull
git push

# 拉取时提示冲突:打开冲突文件,找 <<<<<<< 标记,手动改完保留正确内容后
git add 文件名
git commit
```

> 冲突不可怕:它只是"两个人改了同一处"。改不动就 `git merge --abort` 退出合并,
> 到群里喊人。

## 7. 速查表

| 想做什么 | 命令 |
|---|---|
| 看当前状态 | `git status` |
| 看改动内容 | `git diff` / `git diff --staged` |
| 看历史 | `git log --oneline --graph --all -15` |
| 切分支 | `git switch 分支名` |
| 新建并切换 | `git switch -c 新分支名` |
| 把 main 最新合进自己分支 | `git switch huang && git merge main` |
| 推自己分支(第一次,设上游) | `git push -u origin huang` |
| 日常推送(已设过上游) | `git push` |
| 撤销最近一次 commit(保留改动) | `git reset --soft HEAD~1` |
| 丢弃某文件未提交的改动 | `git restore 文件名` |
