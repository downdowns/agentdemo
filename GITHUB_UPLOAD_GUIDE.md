# GitHub 上传指南

这份文档记录本项目如何上传到 GitHub，避免下次重复查。

---

## 1. 上传前先检查什么

上传前先确认不要把这些文件传到 GitHub：

```text
.env                 # API Key，绝对不能上传
node_modules/         # Node 依赖目录
__pycache__/          # Python 缓存
*.pyc                 # Python 编译缓存
logs/                 # 本地运行日志
chroma_db/            # 本地 Chroma 向量库，可由 docs/ 重建
.DS_Store             # macOS 系统文件
```

本项目已经在 `.gitignore` 中忽略了这些内容。

---

## 2. 推荐上传的内容

建议上传：

```text
README.md
PROJECT_OVERVIEW.md
TECH_NOTES.md
INTERVIEW_QA.md
GITHUB_UPLOAD_GUIDE.md
.env.example
agent.py
app/main.py
config.py
models.py
schemas.py
tools.py
vector_store.py
RAG_Agent_demo.py
eval/questions.json
eval/run_eval.py
docs/*.md
```

其中：

- `README.md`：给面试官看的项目首页
- `.env.example`：告诉别人需要哪些环境变量，但不暴露真实 Key
- `docs/`：本地知识库原文，建议上传，方便别人复现 RAG
- `chroma_db/`：不建议上传，因为可以根据 `docs/` 自动重建

---

## 3. 如果还没有 GitHub 仓库

### 3.1 在 GitHub 网页创建仓库

1. 打开 GitHub
2. 点击右上角 `+`
3. 选择 `New repository`
4. Repository name 可以写：

```text
enterprise-rag-agent
```

5. 建议选择：

```text
Public
```

6. 不要勾选初始化 README，因为本地已经有 README
7. 创建仓库

创建后 GitHub 会给你一个远程地址，类似：

```text
https://github.com/你的用户名/enterprise-rag-agent.git
```

或者 SSH 地址：

```text
git@github.com:你的用户名/enterprise-rag-agent.git
```

---

## 4. 本地第一次上传

在项目根目录执行。

### 4.1 查看当前状态

```bash
git status
```

### 4.2 如果本地还没有初始化 Git

如果执行 `git status` 提示不是 git 仓库，则执行：

```bash
git init
```

你的项目当前已经是 Git 仓库的话，可以跳过这一步。

---

### 4.3 如果之前不小心跟踪了不该上传的文件

`.gitignore` 只能阻止“未被 Git 跟踪的新文件”。  
如果某些文件以前已经被 Git 跟踪，即使写进 `.gitignore`，Git 仍然会继续跟踪。

本项目中 `chroma_db/` 可能已经被跟踪过。如果你想从 Git 跟踪中移除，但保留本地文件，执行：

```bash
git rm -r --cached chroma_db
```

如果 `logs/` 已经被跟踪过，执行：

```bash
git rm -r --cached logs
```

如果 `.env` 不小心被跟踪过，执行：

```bash
git rm --cached .env
```

注意：这些命令只是不再让 Git 跟踪文件，不会删除你本地文件。

---

### 4.4 添加文件

```bash
git add .
```

然后检查：

```bash
git status
```

重点确认不要出现：

```text
.env
logs/
chroma_db/
node_modules/
__pycache__/
```

如果出现，先不要 commit，回到上一步处理。

---

### 4.5 提交 commit

```bash
git commit -m "Initial commit: enterprise RAG agent"
```

---

### 4.6 关联远程仓库

把下面地址替换成你的 GitHub 仓库地址：

```bash
git remote add origin https://github.com/你的用户名/enterprise-rag-agent.git
```

如果已经添加过 origin，但地址不对，可以改：

```bash
git remote set-url origin https://github.com/你的用户名/enterprise-rag-agent.git
```

查看远程地址：

```bash
git remote -v
```

---

### 4.7 推送到 GitHub

如果当前分支叫 `main`：

```bash
git push -u origin main
```

如果当前分支叫 `master`：

```bash
git push -u origin master
```

查看当前分支：

```bash
git branch --show-current
```

如果你想统一改成 `main`：

```bash
git branch -M main
git push -u origin main
```

---

## 5. 以后每次更新代码怎么上传

以后只需要三步：

```bash
git status
git add .
git commit -m "你的提交说明"
git push
```

例如：

```bash
git add .
git commit -m "Add FastAPI error handling"
git push
```

---

## 6. 常见问题

### Q1：为什么 `.env` 不能上传？

`.env` 里有 API Key。

如果上传到公开 GitHub，别人可能拿你的 Key 调接口，造成费用损失和安全风险。

正确做法是上传 `.env.example`，不上传真实 `.env`。

---

### Q2：为什么不上传 `chroma_db/`？

`chroma_db/` 是本地向量库产物。

它可以由 `docs/` 里的文档重新生成，而且里面通常包含二进制文件，不适合放进 Git。

正确做法：

```text
上传 docs/
忽略 chroma_db/
```

别人运行项目时，程序会重新创建 Chroma 数据库。

---

### Q3：为什么 `.gitignore` 写了，但文件还是出现在 git status？

因为这个文件之前已经被 Git 跟踪了。

解决：

```bash
git rm --cached 文件名
```

目录用：

```bash
git rm -r --cached 目录名
```

---

### Q4：push 时提示没有权限怎么办？

可能原因：

1. GitHub 没登录
2. HTTPS 需要 token，不支持密码
3. SSH key 没配置
4. remote 地址写错

先查看：

```bash
git remote -v
```

如果使用 HTTPS，GitHub 现在通常需要 Personal Access Token。

如果嫌麻烦，可以用 GitHub Desktop 或者配置 SSH。

---

## 7. 本项目推荐的首次上传命令汇总

假设你已经在 GitHub 创建了空仓库：

```bash
# 1. 确认状态
git status

# 2. 如果 chroma_db / logs 曾经被跟踪过，取消跟踪
git rm -r --cached chroma_db
git rm -r --cached logs

# 3. 添加文件
git add .

# 4. 再次检查，确认没有 .env / chroma_db / logs
git status

# 5. 提交
git commit -m "Initial commit: enterprise RAG agent"

# 6. 设置远程仓库，替换成你的地址
git remote add origin https://github.com/你的用户名/enterprise-rag-agent.git

# 7. 推送
git branch -M main
git push -u origin main
```

如果 `git remote add origin` 报错说 origin 已存在，用：

```bash
git remote set-url origin https://github.com/你的用户名/enterprise-rag-agent.git
```

