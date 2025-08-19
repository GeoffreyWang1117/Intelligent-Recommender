#!/usr/bin/env python3
"""
在线推荐系统Flask应用

提供RESTful API接口用于：
- 用户推荐
- 相似物品查找
- 健康检查
- 模型信息查询

简化版本，专注于核心功能。
"""

from flask import Flask, request, jsonify, render_template
from datetime import datetime
import logging
import os
import sys
import traceback
from typing import Dict, List, Any, Optional

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from config import Config
    from models.svd_model import SVDRecommender
    from models.faiss_index import FAISSIndex
except ImportError as e:
    print(f"导入模块失败: {e}")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 全局变量
svd_model: Optional[SVDRecommender] = None
faiss_index: Optional[FAISSIndex] = None
movie_data: Dict[int, Dict[str, str]] = {}


def load_movie_data() -> Dict[int, Dict[str, str]]:
    """
    加载电影数据
    
    Returns:
        Dict[int, Dict[str, str]]: 电影ID到电影信息的映射
    """
    import pandas as pd
    
    try:
        movies_path = os.path.join('data', 'sample', 'movies.csv')
        if os.path.exists(movies_path):
            movies_df = pd.read_csv(movies_path)
            movie_dict = {}
            for _, row in movies_df.iterrows():
                movie_dict[int(row['item_id'])] = {
                    'title': row['title'],
                    'genres': row['genres']
                }
            logger.info("电影数据加载成功: %d 部电影", len(movie_dict))
            return movie_dict
        else:
            logger.warning("电影数据文件不存在: %s", movies_path)
            return {}
    except Exception as e:
        logger.error("加载电影数据失败: %s", str(e))
        return {}


def enrich_recommendations(recommendations: List[tuple]) -> List[Dict[str, Any]]:
    """
    为推荐结果添加电影详细信息
    
    Args:
        recommendations: 推荐列表 [(item_id, score), ...]
        
    Returns:
        List[Dict[str, Any]]: 包含详细信息的推荐列表
    """
    enriched = []
    for item_id, score in recommendations:
        movie_info = movie_data.get(item_id, {})
        enriched.append({
            'item_id': item_id,
            'score': score,
            'title': movie_info.get('title', f'电影 #{item_id}'),
            'genres': movie_info.get('genres', '未知类型'),
            'year': extract_year_from_title(movie_info.get('title', '')),
            'genre_list': movie_info.get('genres', '').split('|') if movie_info.get('genres') else []
        })
    return enriched


def extract_year_from_title(title: str) -> Optional[int]:
    """
    从电影标题中提取年份
    
    Args:
        title: 电影标题
        
    Returns:
        Optional[int]: 年份，如果提取失败返回None
    """
    import re
    match = re.search(r'\((\d{4})\)', title)
    return int(match.group(1)) if match else None


def initialize_services() -> bool:
    """
    初始化服务组件
    
    Returns:
        bool: 是否初始化成功
    """
    global svd_model, faiss_index, movie_data
    
    try:
        logger.info("开始初始化服务...")
        
        # 加载电影数据
        movie_data = load_movie_data()
        
        # 初始化SVD模型
        svd_model = SVDRecommender(n_components=50)
        
        # 加载已训练的模型
        model_path = os.path.join('models', 'saved', 'svd_model.pkl')
        if os.path.exists(model_path):
            svd_model.load_model(model_path)
            logger.info("SVD模型加载成功")
        else:
            logger.warning("SVD模型文件不存在: %s", model_path)
            return False
        
        # 初始化FAISS索引
        faiss_index = FAISSIndex(
            dimension=50,
            nlist=min(100, 10)  # 默认值
        )
        
        # 加载FAISS索引
        faiss_path = os.path.join('models', 'saved', 'faiss_index.bin')
        if os.path.exists(faiss_path):
            faiss_index.load_index(faiss_path)
            logger.info("FAISS索引加载成功")
        else:
            logger.warning("FAISS索引文件不存在: %s", faiss_path)
            return False
        
        logger.info("服务初始化完成")
        return True
        
    except Exception as e:
        logger.error("服务初始化失败: %s", str(e))
        logger.error("详细错误: %s", traceback.format_exc())
        return False


@app.route('/', methods=['GET'])
def index() -> str:
    """
    主页 - 显示推荐系统界面
    
    Returns:
        str: 渲染的HTML页面
    """
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health_check() -> tuple:
    """
    健康检查接口
    
    Returns:
        tuple: (JSON响应, HTTP状态码)
    """
    try:
        if svd_model is None or faiss_index is None:
            return jsonify({
                'status': 'unhealthy',
                'message': '服务未初始化',
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }), 503
        
        # 简单的健康检查
        health_info = {
            'status': 'healthy',
            'services': {
                'svd_model': svd_model is not None,
                'faiss_index': faiss_index is not None
            },
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        
        return jsonify(health_info), 200
        
    except Exception as e:
        logger.error("健康检查失败: %s", str(e))
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }), 500


@app.route('/recommend', methods=['GET'])
def get_recommendations() -> tuple:
    """
    获取用户推荐
    
    Args (URL参数):
        user_id: 用户ID (必需)
        top_k: 推荐数量 (可选，默认10)
    
    Returns:
        tuple: (JSON响应, HTTP状态码)
    """
    try:
        if svd_model is None:
            return jsonify({'error': '推荐服务未初始化'}), 503
        
        # 获取参数
        user_id = request.args.get('user_id', type=int)
        top_k = request.args.get('top_k', default=10, type=int)
        
        # 参数验证
        if user_id is None:
            return jsonify({'error': 'user_id参数必须提供'}), 400
        
        if user_id <= 0:
            return jsonify({'error': 'user_id必须为正整数'}), 400
        
        if top_k <= 0 or top_k > 100:
            return jsonify({'error': 'top_k必须在1-100之间'}), 400
        
        # 获取推荐
        recommendations = svd_model.get_user_recommendations(user_id, top_k)
        
        # 丰富推荐结果
        enriched_recommendations = enrich_recommendations(recommendations)
        
        response = {
            'user_id': user_id,
            'recommendations': enriched_recommendations,
            'count': len(enriched_recommendations),
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error("获取推荐失败: %s", str(e))
        return jsonify({
            'error': '内部服务器错误',
            'message': str(e),
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }), 500


@app.route('/similar', methods=['GET'])
def get_similar_items() -> tuple:
    """
    获取相似物品
    
    Args (URL参数):
        item_id: 物品ID (必需)
        top_k: 相似物品数量 (可选，默认10)
    
    Returns:
        tuple: (JSON响应, HTTP状态码)
    """
    try:
        if faiss_index is None or svd_model is None:
            return jsonify({'error': '相似性服务未初始化'}), 503
        
        # 获取参数
        item_id = request.args.get('item_id', type=int)
        top_k = request.args.get('top_k', default=10, type=int)
        
        # 参数验证
        if item_id is None:
            return jsonify({'error': 'item_id参数必须提供'}), 400
        
        if item_id <= 0:
            return jsonify({'error': 'item_id必须为正整数'}), 400
        
        if top_k <= 0 or top_k > 100:
            return jsonify({'error': 'top_k必须在1-100之间'}), 400
        
        # 获取相似物品
        similar_items = svd_model.get_similar_items(item_id, top_k)
        
        # 丰富相似物品结果
        enriched_similar_items = enrich_recommendations(similar_items)
        
        response = {
            'item_id': item_id,
            'item_info': movie_data.get(item_id, {'title': f'电影 #{item_id}', 'genres': '未知类型'}),
            'similar_items': enriched_similar_items,
            'count': len(enriched_similar_items),
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error("获取相似物品失败: %s", str(e))
        return jsonify({
            'error': '内部服务器错误',
            'message': str(e),
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }), 500


@app.route('/info', methods=['GET'])
def get_model_info() -> tuple:
    """
    获取模型信息
    
    Returns:
        tuple: (JSON响应, HTTP状态码)
    """
    try:
        if svd_model is None:
            return jsonify({'error': '模型未初始化'}), 503
        
        # 获取模型基本信息
        model_info = {
            'model_type': 'SVD',
            'n_components': getattr(svd_model, 'n_components', 'unknown'),
            'n_users': len(getattr(svd_model, 'user_encoder', {}).classes_) if hasattr(svd_model, 'user_encoder') else 0,
            'n_items': len(getattr(svd_model, 'item_encoder', {}).classes_) if hasattr(svd_model, 'item_encoder') else 0,
            'has_faiss_index': faiss_index is not None,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        
        return jsonify(model_info), 200
        
    except Exception as e:
        logger.error("获取模型信息失败: %s", str(e))
        return jsonify({
            'error': '内部服务器错误',
            'message': str(e),
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }), 500


@app.errorhandler(404)
def not_found(error) -> tuple:
    """404错误处理"""
    return jsonify({
        'error': '接口不存在',
        'message': '请检查请求URL',
        'available_endpoints': [
            '/ [GET] - 主页界面',
            '/health [GET] - 健康检查',
            '/recommend [GET] - 用户推荐',
            '/similar [GET] - 相似物品',
            '/info [GET] - 模型信息'
        ],
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }), 404


@app.errorhandler(405)
def method_not_allowed(error) -> tuple:
    """405错误处理"""
    return jsonify({
        'error': '请求方法不允许',
        'message': '请检查HTTP方法',
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }), 405


@app.errorhandler(500)
def internal_error(error) -> tuple:
    """500错误处理"""
    return jsonify({
        'error': '内部服务器错误',
        'message': '服务器处理请求时发生错误',
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }), 500


if __name__ == '__main__':
    # 初始化服务
    if not initialize_services():
        logger.error("应用启动失败：服务初始化失败")
        sys.exit(1)
    
    # 启动Flask应用
    try:
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False
        )
    except Exception as e:
        logger.error("启动Flask应用失败: %s", str(e))
        sys.exit(1)
