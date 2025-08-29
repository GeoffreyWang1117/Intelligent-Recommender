#!/usr/bin/env python3
"""
Amazon Reviews 2023数据集处理和LLM推荐验证系统
Amazon Reviews 2023 Data Integration for LLM Recommendation Validation

功能:
1. 通过Hugging Face下载真实Amazon Reviews 2023数据集
2. 构建用户画像和商品特征
3. LLM推荐生成和验证
4. 推荐系统标准评价指标验证

作者: GitHub Copilot
日期: 2025-08-27
"""

import pandas as pd
import numpy as np
import requests
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径，以便导入数据下载器
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

# 导入新的数据下载器
try:
    from data_processing.amazon_downloader import AmazonReviewsDownloader
    DOWNLOADER_AVAILABLE = True
except ImportError:
    DOWNLOADER_AVAILABLE = False
    print("⚠️  数据下载器不可用")

class AmazonReviewsLLMValidator:
    """基于真实Amazon Reviews 2023数据的LLM推荐验证器"""
    
    def __init__(self, data_path: Optional[str] = None, ollama_endpoint: str = "http://localhost:11434/api/generate"):
        # 确保使用项目根目录下的data/amazon路径
        if data_path is None:
            # 获取项目根目录（从当前文件向上两级）
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent
            self.data_path = project_root / "data" / "amazon"
        else:
            self.data_path = Path(data_path)
        
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.ollama_endpoint = ollama_endpoint
        
        # 初始化下载器
        self.downloader = AmazonReviewsDownloader() if DOWNLOADER_AVAILABLE else None
        
        # LLM模型配置
        self.primary_model = "llama3:latest"    # 英文主力
        self.secondary_model = "qwen3:latest"   # 中文对照
        
        # 数据存储
        self.reviews = None
        self.items = None
        self.users = None
        self.user_profiles = {}
        self.item_features = {}
        
        # Amazon Reviews 2023支持的类别 (Hugging Face)
        self.supported_categories = [
            "All_Beauty",
            "Amazon_Fashion", 
            "Appliances",
            "Arts_Crafts_and_Sewing",
            "Automotive",
            "Baby_Products",
            "Beauty_and_Personal_Care",
            "Books",
            "CDs_and_Vinyl",
            "Cell_Phones_and_Accessories", 
            "Clothing_Shoes_and_Jewelry",
            "Digital_Music",
            "Electronics",
            "Gift_Cards",
            "Grocery_and_Gourmet_Food",
            "Handmade_Products",
            "Health_and_Household",
            "Health_and_Personal_Care",
            "Home_and_Garden", 
            "Home_and_Kitchen",
            "Industrial_and_Scientific",
            "Kindle_Store",
            "Magazine_Subscriptions",
            "Movies_and_TV",
            "Musical_Instruments",
            "Office_Products",
            "Patio_Lawn_and_Garden",
            "Pet_Supplies",
            "Software",
            "Sports_and_Outdoors",
            "Subscription_Boxes",
            "Tools_and_Home_Improvement",
            "Toys_and_Games",
            "Video_Games"
        ]
        
    def download_and_load_amazon_data(self, category: str = "Movies_and_TV", sample_size: int = 50000) -> bool:
        """
        通过Hugging Face下载并加载真实Amazon Reviews 2023完整数据集
        
        Args:
            category: 商品类别
            sample_size: 采样大小（增大以获取更多真实数据）
        """
        if not DATASETS_AVAILABLE:
            print("❌ datasets库未安装，请运行: pip install datasets")
            return False
        
        if category not in self.supported_categories:
            print(f"❌ 不支持的类别: {category}")
            print(f"支持的类别: {self.supported_categories[:10]}...") # 显示前10个
            return False
        
        # 检查本地缓存
        reviews_file = self.data_path / f"{category}_reviews.parquet"
        items_file = self.data_path / f"{category}_items.parquet"
        
        if reviews_file.exists() and items_file.exists():
            print(f"✅ 发现本地缓存文件，直接加载...")
            try:
                self.reviews = pd.read_parquet(reviews_file)
                self.items = pd.read_parquet(items_file)
                
                # 构建用户数据
                user_stats = self.reviews.groupby('user_id').agg({
                    'rating': ['count', 'mean', 'std'],
                    'timestamp': ['min', 'max']
                }).reset_index()
                user_stats.columns = ['user_id', 'review_count', 'avg_rating', 'rating_std', 'first_review', 'last_review']
                self.users = user_stats
                
                print(f"✅ 从缓存加载成功:")
                print(f"   📊 评论数: {len(self.reviews):,}")
                print(f"   👥 用户数: {len(self.users):,}")
                print(f"   🛍️  商品数: {len(self.items):,}")
                return True
                
            except Exception as e:
                print(f"⚠️  缓存文件损坏，重新下载: {e}")
        
        try:
            print(f"🔽 从Hugging Face下载Amazon Reviews 2023完整数据集: {category}")
            print(f"   目标采样数: {sample_size:,} 条评论")
            
            # 下载评论数据
            print("   🔄 加载评论数据（流式处理）...")
            try:
                # 使用Hugging Face datasets库下载数据
                dataset_name = f"McAuley-Lab/Amazon-Reviews-2023"
                
                # 尝试不同的数据集配置名称
                possible_names = [
                    f"raw_review_{category}",
                    f"review_{category}", 
                    category
                ]
                
                reviews_dataset = None
                for name in possible_names:
                    try:
                        reviews_dataset = load_dataset(
                            dataset_name, 
                            name=name,
                            split="full",
                            streaming=True,
                            trust_remote_code=True
                        )
                        print(f"   ✅ 使用配置名称: {name}")
                        break
                    except Exception as e:
                        print(f"   ⚠️  尝试配置 {name} 失败: {str(e)[:100]}...")
                        continue
                
                if reviews_dataset is None:
                    print(f"❌ 所有配置都失败，无法加载 {category} 数据")
                    return False
                
                # 采样数据 - 确保获取足够的真实数据
                reviews_data = []
                print(f"   📊 开始采样 {sample_size:,} 条评论...")
                
                batch_size = 5000
                processed = 0
                
                for i, review in enumerate(reviews_dataset):
                    if i >= sample_size:
                        break
                    
                    # 处理评论数据 - 提取所有可用字段
                    review_item = {
                        'user_id': str(review.get('user_id', review.get('reviewerID', ''))),
                        'item_id': str(review.get('asin', review.get('parent_asin', ''))),
                        'rating': float(review.get('rating', review.get('overall', 0))),
                        'timestamp': review.get('timestamp', review.get('unixReviewTime', 0)),
                        'verified_purchase': review.get('verified_purchase', False),
                        'helpful_vote': review.get('helpful_vote', 0),
                        'text': str(review.get('text', review.get('reviewText', '')))[:1000]  # 保留更多文本
                    }
                    
                    # 只保留有效的评论
                    if (review_item['user_id'] and 
                        review_item['item_id'] and 
                        review_item['rating'] > 0):
                        reviews_data.append(review_item)
                        processed += 1
                    
                    if (i + 1) % batch_size == 0:
                        print(f"     处理进度: {i+1:,}/{sample_size:,} ({(i+1)/sample_size*100:.1f}%), 有效评论: {processed:,}")
                
                if not reviews_data:
                    print("❌ 没有获取到有效的评论数据")
                    return False
                
                self.reviews = pd.DataFrame(reviews_data)
                print(f"   ✅ 成功加载 {len(self.reviews):,} 条真实评论")
                print(f"   📈 数据质量: 用户数 {self.reviews['user_id'].nunique():,}, 商品数 {self.reviews['item_id'].nunique():,}")
                
            except Exception as e:
                print(f"❌ 评论数据下载失败: {e}")
                import traceback
                traceback.print_exc()
                return False
            
            # 下载商品元数据
            print("   🔄 加载商品元数据...")
            try:
                # 尝试不同的元数据配置
                meta_names = [
                    f"raw_meta_{category}",
                    f"meta_{category}",
                    f"{category}_meta"
                ]
                
                meta_dataset = None
                for name in meta_names:
                    try:
                        meta_dataset = load_dataset(
                            dataset_name,
                            name=name,
                            split="full",
                            streaming=True,
                            trust_remote_code=True
                        )
                        print(f"   ✅ 使用元数据配置: {name}")
                        break
                    except Exception as e:
                        print(f"   ⚠️  尝试元数据配置 {name} 失败: {str(e)[:100]}...")
                        continue
                
                if meta_dataset is None:
                    print("⚠️  无法加载元数据，将创建基础商品信息")
                    self._create_basic_item_info()
                else:
                    # 只保留评论中出现的商品
                    item_ids = set(self.reviews['item_id'].unique())
                    items_data = []
                    
                    print(f"   🔍 匹配 {len(item_ids):,} 个商品的元数据...")
                    matched_count = 0
                    processed_meta = 0
                    
                    for i, item in enumerate(meta_dataset):
                        processed_meta += 1
                        asin = str(item.get('asin', item.get('parent_asin', '')))
                        
                        if asin in item_ids:
                            # 提取完整的商品信息
                            item_info = {
                                'item_id': asin,
                                'title': str(item.get('title', ''))[:200],
                                'brand': str(item.get('brand', '')),
                                'category': item.get('category', item.get('categories', []))[:10],  # 保留更多类别
                                'price': str(item.get('price', '')),
                                'description': str(item.get('description', []))[:500],
                                'features': item.get('features', item.get('feature', []))[:10],  # 保留更多特征
                                'image_url': item.get('imageURL', item.get('images', [])),
                                'store': item.get('store', ''),
                                'main_category': item.get('main_category', '')
                            }
                            
                            items_data.append(item_info)
                            matched_count += 1
                            
                            if matched_count % 1000 == 0:
                                print(f"     匹配进度: {matched_count:,}/{len(item_ids):,} ({matched_count/len(item_ids)*100:.1f}%)")
                        
                        # 限制元数据处理数量以避免无限循环
                        if processed_meta > len(item_ids) * 3 or matched_count >= len(item_ids) * 0.9:
                            break
                    
                    self.items = pd.DataFrame(items_data)
                    print(f"   ✅ 成功匹配 {len(self.items):,} 个商品 (匹配率: {len(self.items)/len(item_ids)*100:.1f}%)")
                    
                    # 为未匹配的商品创建基础信息
                    if len(self.items) < len(item_ids):
                        missing_items = item_ids - set(self.items['item_id'])
                        print(f"   📝 为 {len(missing_items):,} 个缺失商品创建基础信息...")
                        
                        basic_items = pd.DataFrame({
                            'item_id': list(missing_items),
                            'title': [f"{category.replace('_', ' ')} Product {item[-6:]}" for item in missing_items],
                            'brand': ['Unknown'] * len(missing_items),
                            'category': [[category.replace('_', ' ')]] * len(missing_items),
                            'main_category': [category.replace('_', ' ')] * len(missing_items)
                        })
                        
                        self.items = pd.concat([self.items, basic_items], ignore_index=True)
                        print(f"   � 总商品数: {len(self.items):,}")
                
            except Exception as e:
                print(f"⚠️  商品元数据加载失败: {e}")
                self._create_basic_item_info()
            
            # 构建用户数据
            print("   👥 构建用户档案...")
            user_stats = self.reviews.groupby('user_id').agg({
                'rating': ['count', 'mean', 'std'],
                'timestamp': ['min', 'max'],
                'verified_purchase': 'mean'
            }).reset_index()
            
            user_stats.columns = ['user_id', 'review_count', 'avg_rating', 'rating_std', 'first_review', 'last_review', 'verified_rate']
            self.users = user_stats
            print(f"   ✅ 构建了 {len(self.users):,} 个用户档案")
            
            # 轻量级数据清洗 - 只移除明显无效的数据
            self._minimal_clean_data()
            
            # 保存到本地缓存
            try:
                print(f"   💾 保存数据到本地缓存...")
                reviews_file = self.data_path / f"{category}_reviews.parquet"
                items_file = self.data_path / f"{category}_items.parquet"
                
                self.reviews.to_parquet(reviews_file, index=False)
                self.items.to_parquet(items_file, index=False)
                print(f"   ✅ 数据已保存到 {self.data_path}")
            except Exception as e:
                print(f"   ⚠️  保存缓存失败: {e}")
            
            print(f"🎉 Amazon Reviews真实数据加载完成:")
            print(f"   类别: {category}")
            print(f"   评论数: {len(self.reviews):,}")
            print(f"   用户数: {len(self.users):,}")
            print(f"   商品数: {len(self.items):,}")
            print(f"   评分分布: {dict(self.reviews['rating'].value_counts().sort_index())}")
            print(f"   数据密度: {len(self.reviews)/(len(self.users)*len(self.items))*100:.4f}%")
            
            return True
            
        except Exception as e:
            print(f"❌ 真实数据下载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_basic_item_info(self):
        """为商品创建基础信息"""
        unique_items = self.reviews['item_id'].unique()
        self.items = pd.DataFrame({
            'item_id': unique_items,
            'title': [f"Product {item[-6:]}" for item in unique_items],
            'brand': ['Unknown'] * len(unique_items),
            'category': [['Unknown']] * len(unique_items),
            'main_category': ['Unknown'] * len(unique_items)
        })
        print(f"   📝 创建了 {len(self.items):,} 个基础商品信息")
    
    def _minimal_clean_data(self):
        """轻量级数据清洗 - 保留更多真实数据"""
        print("🧹 轻量级数据清洗...")
        
        if self.reviews is None or len(self.reviews) == 0:
            print("   ❌ 没有评论数据需要清洗")
            return
        
        initial_reviews = len(self.reviews)
        
        # 只移除明显无效的数据
        self.reviews = self.reviews.dropna(subset=['user_id', 'item_id', 'rating'])
        self.reviews = self.reviews[self.reviews['rating'] > 0]
        self.reviews = self.reviews[self.reviews['rating'] <= 5]  # 确保评分在1-5范围内
        self.reviews = self.reviews[self.reviews['user_id'] != '']
        self.reviews = self.reviews[self.reviews['item_id'] != '']
        
        # 非常宽松的过滤 - 只移除明显的异常数据
        user_counts = self.reviews['user_id'].value_counts()
        item_counts = self.reviews['item_id'].value_counts()
        
        # 只移除只有1个评论的极端情况
        active_users = user_counts[user_counts >= 1].index
        popular_items = item_counts[item_counts >= 1].index
        
        self.reviews = self.reviews[
            (self.reviews['user_id'].isin(active_users)) & 
            (self.reviews['item_id'].isin(popular_items))
        ]
        
        print(f"   清洗前: {initial_reviews:,} 条评论")
        print(f"   清洗后: {len(self.reviews):,} 条评论")
        print(f"   保留率: {len(self.reviews)/initial_reviews*100:.1f}%")
    
    def load_amazon_data(self, category: str = "Movies_and_TV", sample_size: int = 50000) -> bool:
        """
        加载Amazon Reviews数据集 - 只使用真实数据，不生成假数据
        
        Args:
            category: 商品类别
            sample_size: 采样大小（增大以获取更多真实数据）
        """
        print(f"📊 加载Amazon Reviews真实数据: {category}")
        print(f"⚠️  注意: 此方法只使用真实数据，不会生成假数据")
        
        # 只尝试Hugging Face真实数据
        success = self.download_and_load_amazon_data(category, sample_size)
        
        if not success:
            print("❌ 真实数据加载失败")
            print("💡 建议:")
            print("   1. 检查网络连接")
            print("   2. 尝试不同的category名称")
            print("   3. 减少sample_size")
            print(f"   4. 支持的类别: {self.supported_categories[:5]}...")
            return False
        
        if self.reviews is None or len(self.reviews) == 0:
            print("❌ 加载的数据为空")
            return False
        
        print(f"✅ 真实Amazon Reviews数据加载成功:")
        print(f"   📊 数据规模: {len(self.reviews):,} 条评论")
        print(f"   👥 用户规模: {len(self.users):,} 个用户") 
        print(f"   🛍️  商品规模: {len(self.items):,} 个商品")
        
        return True

    def _clean_data(self):
        """数据清洗"""
        print("🧹 数据清洗...")
        
        if self.reviews is None or len(self.reviews) == 0:
            print("   ❌ 没有评论数据需要清洗")
            return
        
        # 移除空值
        initial_reviews = len(self.reviews)
        self.reviews = self.reviews.dropna(subset=['user_id', 'item_id', 'rating'])
        self.reviews = self.reviews[self.reviews['rating'] > 0]  # 移除无效评分
        
        # 放宽过滤条件 - 过滤极少活跃的用户和商品
        user_counts = self.reviews['user_id'].value_counts()
        item_counts = self.reviews['item_id'].value_counts()
        
        active_users = user_counts[user_counts >= 2].index  # 至少2个评论
        popular_items = item_counts[item_counts >= 1].index  # 至少1个评论
        
        self.reviews = self.reviews[
            (self.reviews['user_id'].isin(active_users)) & 
            (self.reviews['item_id'].isin(popular_items))
        ]
        
        print(f"   清洗前: {initial_reviews} 条评论")
        print(f"   清洗后: {len(self.reviews)} 条评论")

    def build_user_profiles(self) -> Dict:
        """构建用户画像"""
        print("👤 构建用户画像...")
        
        self.user_profiles = {}
        
        for user_id in self.users['user_id'].unique():
            user_info = self.users[self.users['user_id'] == user_id].iloc[0]
            user_reviews = self.reviews[self.reviews['user_id'] == user_id]
            
            # 基础统计信息
            profile = {
                'user_id': user_id,
                'total_ratings': len(user_reviews),
                'avg_rating': user_reviews['rating'].mean(),
                'rating_std': user_reviews['rating'].std(),
                'purchase_history': user_reviews['item_id'].tolist(),
                'preferred_categories': [],
                'brand_preferences': {},
                'rating_distribution': user_reviews['rating'].value_counts().to_dict()
            }
            
            # 商品偏好分析
            if len(user_reviews) > 0 and hasattr(self, 'items') and self.items is not None:
                user_items = self.items[self.items['item_id'].isin(user_reviews['item_id'])]
                
                # 品牌偏好
                if 'brand' in user_items.columns:
                    brand_ratings = {}
                    for _, item in user_items.iterrows():
                        brand = item['brand']
                        item_rating = user_reviews[user_reviews['item_id'] == item['item_id']]['rating'].iloc[0]
                        if brand not in brand_ratings:
                            brand_ratings[brand] = []
                        brand_ratings[brand].append(item_rating)
                    
                    # 计算品牌平均评分
                    profile['brand_preferences'] = {
                        brand: np.mean(ratings) for brand, ratings in brand_ratings.items()
                    }
                
                # 类别偏好
                if 'category' in user_items.columns:
                    categories = []
                    for _, item in user_items.iterrows():
                        if isinstance(item['category'], list):
                            categories.extend(item['category'])
                        elif isinstance(item['category'], str):
                            categories.append(item['category'])
                    profile['preferred_categories'] = list(set(categories))
            
            self.user_profiles[user_id] = profile
        
        print(f"✅ 成功构建 {len(self.user_profiles)} 个用户画像")
        return self.user_profiles

    def generate_llm_recommendation(self, user_id: str, model: str = "llama3:latest", 
                                  candidate_items: Optional[List[str]] = None, k: int = 10) -> List[str]:
        """为指定用户生成LLM推荐"""
        
        if user_id not in self.user_profiles:
            return []
        
        user_profile = self.user_profiles[user_id]
        
        # 获取候选商品
        if candidate_items is None:
            # 排除用户已购买的商品
            purchased_items = set(user_profile['purchase_history'])
            all_items = set(self.items['item_id'])
            candidate_items = list(all_items - purchased_items)
            
            # 限制候选商品数量
            if len(candidate_items) > 50:
                candidate_items = np.random.choice(candidate_items, 50, replace=False).tolist()
        
        # 构建商品信息
        candidate_info = []
        for item_id in candidate_items[:30]:  # 限制提示长度
            item_data = self.items[self.items['item_id'] == item_id]
            if not item_data.empty:
                item = item_data.iloc[0]
                candidate_info.append({
                    'id': item_id,
                    'title': item['title'][:100],  # 限制长度
                    'brand': item['brand'],
                    'category': item['category'][:3] if isinstance(item['category'], list) else []
                })
        
        # 构建LLM提示
        prompt = self._build_amazon_prompt(user_profile, candidate_info, k)
        
        try:
            # 调用LLM
            response = requests.post(
                self.ollama_endpoint,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 500
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                llm_response = response.json().get('response', '')
                recommendations = self._parse_amazon_recommendations(llm_response, [item['id'] for item in candidate_info])
                return recommendations[:k]
            else:
                print(f"❌ LLM调用失败: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"❌ LLM推荐生成异常: {e}")
            return []

    def _build_amazon_prompt(self, user_profile: Dict, candidate_items: List[Dict], k: int) -> str:
        """构建Amazon推荐的LLM提示"""
        
        # 用户偏好总结
        user_summary = f"""用户画像:
- 总评论数: {user_profile['total_ratings']}
- 平均评分: {user_profile['avg_rating']:.2f}
- 偏好品牌: {', '.join(list(user_profile['brand_preferences'].keys())[:5])}
- 偏好类别: {', '.join(user_profile['preferred_categories'][:5])}
- 评分习惯: 高评分({user_profile['rating_distribution'].get(5, 0)}个), 中评分({user_profile['rating_distribution'].get(4, 0)}个)
"""
        
        # 候选商品信息
        items_info = "候选商品:\n"
        for i, item in enumerate(candidate_items[:20], 1):
            items_info += f"{i}. ID: {item['id']}, 标题: {item['title']}, 品牌: {item['brand']}, 类别: {', '.join(item['category'])}\n"
        
        prompt = f"""
你是一个专业的Amazon商品推荐系统。基于用户的历史行为和偏好，从候选商品中推荐最适合的{k}个商品。

{user_summary}

{items_info}

请根据用户的偏好和购买历史，推荐{k}个最合适的商品。只返回商品ID，用逗号分隔，不要其他解释。

推荐商品ID:"""
        
        return prompt

    def _parse_amazon_recommendations(self, llm_response: str, candidate_items: List[str]) -> List[str]:
        """解析LLM返回的Amazon推荐结果"""
        
        recommendations = []
        
        # 尝试从响应中提取商品ID
        lines = llm_response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line:
                # 寻找逗号分隔的ID
                potential_ids = [item.strip() for item in line.split(',')]
                for item_id in potential_ids:
                    # 清理ID格式
                    item_id = item_id.replace('"', '').replace("'", '').strip()
                    if item_id in candidate_items and item_id not in recommendations:
                        recommendations.append(item_id)
        
        return recommendations

def test_amazon_validator():
    """测试Amazon Reviews验证器 - 只使用真实数据"""
    print("🧪 测试Amazon Reviews LLM验证器（真实数据）")
    print("=" * 60)
    
    validator = AmazonReviewsLLMValidator()
    
    # 测试不同的数据集类别 - 从最可能成功的开始
    test_categories = ["Books", "Movies_and_TV", "Electronics", "Amazon_Fashion"]
    
    for category in test_categories:
        print(f"\n📚 测试类别: {category}")
        print("-" * 40)
        
        # 尝试加载真实数据 (使用较大的采样以获得足够数据)
        if validator.load_amazon_data(category, sample_size=10000):
            print(f"✅ {category} 真实数据加载成功")
            
            # 显示数据统计
            print(f"   📊 评论数: {len(validator.reviews):,}")
            print(f"   👥 用户数: {len(validator.users):,}")
            print(f"   🛍️  商品数: {len(validator.items):,}")
            print(f"   ⭐ 评分分布: {dict(validator.reviews['rating'].value_counts().sort_index())}")
            
            # 构建用户画像
            if validator.build_user_profiles():
                print("✅ 用户画像构建成功")
                
                # 选择一个测试用户进行推荐
                if validator.user_profiles:
                    test_user = list(validator.user_profiles.keys())[0]
                    print(f"🎯 测试用户: {test_user}")
                    
                    # 显示用户信息
                    user_profile = validator.user_profiles[test_user]
                    print(f"   📈 用户统计: {user_profile['total_ratings']} 个评论, 平均评分 {user_profile['avg_rating']:.2f}")
                    
                    # 生成推荐
                    recommendations = validator.generate_llm_recommendation(test_user, k=5)
                    print(f"📝 LLM推荐结果: {recommendations}")
                    
                    # 成功测试一个类别后退出
                    print(f"\n🎉 {category} 测试完成！")
                    return
                else:
                    print("❌ 没有用户画像")
            else:
                print("❌ 用户画像构建失败")
        else:
            print(f"❌ {category} 真实数据加载失败")
    
    print("\n❌ 所有类别的真实数据加载都失败了")
    print("💡 建议检查网络连接和Hugging Face访问")

if __name__ == "__main__":
    test_amazon_validator()


