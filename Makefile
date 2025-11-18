# Makefile for Online Recommendation System

.PHONY: help install train test run docker-build docker-run clean test-all test-unit test-integration test-coverage test-quick

# 默认目标
help:
	@echo "在线推荐系统 - 可用命令:"
	@echo ""
	@echo "开发:"
	@echo "  install     - 安装生产依赖"
	@echo "  install-dev - 安装开发依赖（包括测试工具）"
	@echo "  train       - 训练模型"
	@echo "  run         - 启动API服务"
	@echo "  run-prod    - 生产环境启动"
	@echo ""
	@echo "测试:"
	@echo "  test        - 运行所有测试"
	@echo "  test-unit   - 运行单元测试"
	@echo "  test-integration - 运行集成测试"
	@echo "  test-coverage    - 运行测试并生成覆盖率报告"
	@echo "  test-quick  - 运行快速测试（跳过慢测试）"
	@echo "  test-cache  - 运行缓存相关测试"
	@echo "  test-models - 运行模型相关测试"
	@echo "  test-api    - 运行API测试"
	@echo ""
	@echo "代码质量:"
	@echo "  format      - 格式化代码"
	@echo "  lint        - 代码质量检查"
	@echo "  clean       - 清理临时文件"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build - 构建Docker镜像"
	@echo "  docker-run  - 使用Docker运行"
	@echo "  docker-compose - 使用Docker Compose运行"

# 安装生产依赖
install:
	pip install -r requirements-prod.txt
	@echo "✓ 生产依赖安装完成"

# 安装开发依赖
install-dev:
	pip install -r requirements-dev.txt
	@echo "✓ 开发依赖安装完成（包括测试工具）"

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

# 运行所有测试
test:
	@echo "运行所有测试..."
	python run_tests.py --all -v
	@echo "✓ 所有测试完成"

# 运行单元测试
test-unit:
	@echo "运行单元测试..."
	pytest tests/ -m unit -v
	@echo "✓ 单元测试完成"

# 运行集成测试
test-integration:
	@echo "运行集成测试..."
	python run_tests.py --integration -v
	@echo "✓ 集成测试完成"

# 运行测试并生成覆盖率报告
test-coverage:
	@echo "运行测试并生成覆盖率报告..."
	python run_tests.py --coverage
	@echo "✓ 测试完成，覆盖率报告已生成："
	@echo "  HTML: htmlcov/index.html"
	@echo "  Terminal: 见上方输出"

# 运行快速测试
test-quick:
	@echo "运行快速测试（跳过慢测试）..."
	python run_tests.py --quick
	@echo "✓ 快速测试完成"

# 运行缓存测试
test-cache:
	@echo "运行缓存相关测试..."
	pytest tests/test_cache.py -v
	@echo "✓ 缓存测试完成"

# 运行模型测试
test-models:
	@echo "运行模型相关测试..."
	pytest tests/test_algorithms.py -v
	@echo "✓ 模型测试完成"

# 运行API测试
test-api:
	@echo "运行API测试..."
	python tests/test_api.py --url http://localhost:5000
	@echo "✓ API测试完成"

# 启动API服务
run:
	@echo "启动推荐API服务..."
	python app.py

# 生产环境启动
run-prod:
	@echo "生产环境启动推荐API服务..."
	gunicorn -w 4 -b 0.0.0.0:5000 app:app

# 运行集成测试（旧方式，保持兼容）
integration: test-integration

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
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf *.log
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
