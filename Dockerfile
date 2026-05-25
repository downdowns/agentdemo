# 使用官方 Python 3.11 slim 镜像作为基础环境。
# slim 版本比完整 Python 镜像更小，适合部署 FastAPI 服务。
FROM python:3.11-slim

# 设置容器内的工作目录。
# 后续 COPY、RUN、CMD 都会以 /app 为当前目录。
WORKDIR /app

# 避免 Python 生成 .pyc 缓存文件，减少无用文件。
ENV PYTHONDONTWRITEBYTECODE=1

# 让 Python 日志直接输出到终端，方便 docker logs 查看。
ENV PYTHONUNBUFFERED=1

# 先复制 requirements.txt。
# 这样 Docker 可以缓存依赖安装层，代码变动时不用每次重新安装依赖。
COPY requirements.txt .

# 安装 Python 依赖。
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码到容器。
# .dockerignore 中的文件不会被复制。
COPY . .

# FastAPI 默认运行端口。
EXPOSE 8000

# 启动 FastAPI 服务。
# 注意：Docker 中要监听 0.0.0.0，否则宿主机无法访问容器内服务。
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]