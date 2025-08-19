# Makefile for Online Recommendation System

.PHONY: help install train test run docker-build docker-run clean

# 默认目标
help:
	@echo "在线推荐系统 - 可用命令:"
	@echo "  install     - 安装依赖"
	@echo "  train       - 训练模型"
	@echo "  test        - 运行测试"
	@echo "  run         - 启动API服务"
	@echo "  integration - 运行集成测试"
	@echo "  docker-build - 构建Docker镜像"
	@echo "  docker-run  - 使用Docker运行"
	@echo "  docker-compose - 使用Docker Compose运行"
	@echo "  clean       - 清理临时文件"

# 安装依赖
install:
	pip install -r requirements.txt
	@echo "✓ 依赖安装完成"

# 准备数据
prepare-data:
	@echo "准备MovieLens数据..."
	mkdir -p data
	@if [ ! -f "data/ml-1m.zip" ]; then \
		echo "正在下载MovieLens 1M数据集..."; \
		cd data && wget https://files.grouplens.org/datasets/movielens/ml-1m.zip; \
		unzip ml-1m.zip; \
	else \
		echo "✓ 数据集已存在"; \
	fi

# 训练模型
train: prepare-data
	@echo "开始训练推荐模型..."
	python scripts/train_model.py
	@echo "✓ 模型训练完成"

# 运行单元测试
test:
	@echo "运行单元测试..."
	python -m pytest tests/ -v
	@echo "✓ 单元测试完成"

# 启动API服务
run:
	@echo "启动推荐API服务..."
	python app.py

# 生产环境启动
run-prod:
	@echo "生产环境启动推荐API服务..."
	gunicorn -w 4 -b 0.0.0.0:5000 app:app

# 运行集成测试
integration:
	@echo "运行集成测试..."
	python scripts/test_integration.py
	@echo "✓ 集成测试完成"

# 构建Docker镜像
docker-build:
	@echo "构建Docker镜像..."
	docker build -t recommendation-system .
	@echo "✓ Docker镜像构建完成"

# 运行Docker容器
docker-run: docker-build
	@echo "启动Docker容器..."
	docker run -d --name redis -p 6379:6379 redis:latest
	docker run -d --name rec-system --link redis:redis -p 5000:5000 -e REDIS_HOST=redis recommendation-system
	@echo "✓ Docker容器已启动"

# 使用Docker Compose运行
docker-compose:
	@echo "使用Docker Compose启动服务..."
	docker-compose up -d
	@echo "✓ 服务已启动，访问 http://localhost:5000"

# 停止Docker Compose服务
docker-compose-down:
	@echo "停止Docker Compose服务..."
	docker-compose down
	@echo "✓ 服务已停止"

# 查看服务状态
status:
	@echo "检查服务状态..."
	curl -f http://localhost:5000/health || echo "API服务未响应"
	docker ps | grep -E "(redis|rec-system)" || echo "Docker容器未运行"

# 查看日志
logs:
	@echo "查看应用日志..."
	docker-compose logs -f recommendation-api

# 清理临时文件
clean:
	@echo "清理临时文件..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	@echo "✓ 清理完成"

# 开发环境设置
dev-setup: install prepare-data train
	@echo "✓ 开发环境设置完成"

# 完整部署流程
deploy: docker-build docker-compose
	@echo "等待服务启动..."
	sleep 10
	make integration
	@echo "✓ 部署完成"

# 性能测试
benchmark:
	@echo "运行性能基准测试..."
	python scripts/benchmark.py
	@echo "✓ 性能测试完成"

# 代码格式化
format:
	@echo "格式化代码..."
	black .
	isort .
	@echo "✓ 代码格式化完成"

# 代码检查
lint:
	@echo "代码质量检查..."
	flake8 .
	pylint **/*.py
	@echo "✓ 代码检查完成"
