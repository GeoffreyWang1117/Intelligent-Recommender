#!/usr/bin/env python3
"""
真实MovieLens数据下载器 V2
下载并处理官方MovieLens数据集，按大小分类存储到data/movielens/{size}/

功能:
1. 下载真实MovieLens数据集 (small, 100k, 1m)
2. 清理和标准化数据格式
3. 按数据集大小存储到对应子目录
4. 生成数据统计报告

作者: GitHub Copilot
日期: 2025-08-27
"""

import os
import sys
import requests
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import shutil
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MovieLensDataDownloader:
    """MovieLens数据下载器 - 按大小分类存储"""
    
    def __init__(self, base_dir: str = "../data/movielens"):
        self.base_dir = Path(base_dir)
        self.temp_dir = Path("temp_movielens_download")
        
        # 数据集URL和配置
        self.datasets = {
            "small": {
                "url": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
                "description": "MovieLens Latest Small (9,000 movies, 100,000 ratings)",
                "extract_folder": "ml-latest-small",
                "target_subdir": "small"
            },
            "100k": {
                "url": "https://files.grouplens.org/datasets/movielens/ml-100k.zip", 
                "description": "MovieLens 100K (1,700 movies, 100,000 ratings)",
                "extract_folder": "ml-100k",
                "target_subdir": "100k"
            },
            "1m": {
                "url": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
                "description": "MovieLens 1M (4,000 movies, 1 million ratings)",
                "extract_folder": "ml-1m",
                "target_subdir": "1m"
            }
        }
        
    def download_dataset(self, dataset_name: str = "small", force_download: bool = False) -> bool:
        """下载指定数据集到对应子目录"""
        if dataset_name not in self.datasets:
            logger.error(f"不支持的数据集: {dataset_name}")
            logger.info(f"支持的数据集: {list(self.datasets.keys())}")
            return False
            
        dataset_info = self.datasets[dataset_name]
        target_dir = self.base_dir / dataset_info["target_subdir"]
        
        # 检查是否需要下载
        if not force_download and self._check_existing_data(target_dir):
            logger.info(f"检测到已存在的数据在 {target_dir}，跳过下载。使用 force_download=True 强制重新下载")
            return True
            
        logger.info(f"开始下载: {dataset_info['description']}")
        logger.info(f"目标目录: {target_dir}")
        
        try:
            # 创建临时目录
            self.temp_dir.mkdir(exist_ok=True)
            
            # 下载数据
            zip_path = self.temp_dir / f"{dataset_name}.zip"
            if not self._download_file(dataset_info["url"], zip_path):
                return False
                
            # 解压数据
            extract_path = self.temp_dir / dataset_info["extract_folder"]
            if not self._extract_zip(zip_path, extract_path):
                return False
                
            # 处理和转换数据
            if not self._process_data(extract_path, dataset_name, target_dir):
                return False
                
            # 清理临时文件
            self._cleanup_temp_files()
            
            logger.info(f"数据下载和处理完成，保存到: {target_dir}")
            return True
            
        except Exception as e:
            logger.error(f"下载过程中出现错误: {str(e)}")
            self._cleanup_temp_files()
            return False
    
    def _check_existing_data(self, target_dir: Path) -> bool:
        """检查是否已有数据"""
        required_files = ["movies.csv", "ratings.csv"]
        return all((target_dir / f).exists() for f in required_files)
    
    def _download_file(self, url: str, target_path: Path) -> bool:
        """下载文件"""
        try:
            logger.info("正在下载数据...")
            
            # 禁用SSL验证以解决证书过期问题
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            response = requests.get(url, stream=True, verify=False)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        if total_size > 0:
                            progress = downloaded_size / total_size * 100
                            print(f"\r下载进度: {progress:.1f}% ({downloaded_size:,}/{total_size:,} bytes)", 
                                  end="", flush=True)
                                  
            print()  # 换行
            logger.info("下载完成")
            return True
            
        except Exception as e:
            logger.error(f"下载失败: {str(e)}")
            return False
    
    def _extract_zip(self, zip_path: Path, extract_path: Path) -> bool:
        """解压ZIP文件"""
        try:
            logger.info("正在解压文件...")
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.temp_dir)
                
            logger.info("解压完成")
            return True
            
        except Exception as e:
            logger.error(f"解压失败: {str(e)}")
            return False
    
    def _process_data(self, extract_path: Path, dataset_name: str, target_dir: Path) -> bool:
        """处理数据并保存到目标目录"""
        try:
            logger.info("正在处理数据...")
            
            # 创建目标目录
            target_dir.mkdir(parents=True, exist_ok=True)
            
            if dataset_name == "small":
                return self._process_small_dataset(extract_path, target_dir)
            elif dataset_name == "100k":
                return self._process_100k_dataset(extract_path, target_dir)
            elif dataset_name == "1m":
                return self._process_1m_dataset(extract_path, target_dir)
            else:
                logger.error(f"未知数据集: {dataset_name}")
                return False
                
        except Exception as e:
            logger.error(f"数据处理失败: {str(e)}")
            return False
    
    def _process_small_dataset(self, extract_path: Path, target_dir: Path) -> bool:
        """处理small数据集"""
        try:
            # 处理电影数据
            movies_df = pd.read_csv(extract_path / "movies.csv")
            
            # 分割genres列
            if 'genres' in movies_df.columns:
                movies_df['genres'] = movies_df['genres'].str.replace('|', ', ')
                
            movies_df.to_csv(target_dir / "movies.csv", index=False)
            logger.info(f"处理电影数据: {len(movies_df)} 部电影")
            
            # 处理评分数据
            ratings_df = pd.read_csv(extract_path / "ratings.csv")
            ratings_df.to_csv(target_dir / "ratings.csv", index=False)
            logger.info(f"处理评分数据: {len(ratings_df)} 条评分")
            
            # 复制其他文件（如果存在）
            for filename in ["tags.csv", "links.csv"]:
                source_file = extract_path / filename
                if source_file.exists():
                    shutil.copy2(source_file, target_dir / filename)
                    logger.info(f"复制文件: {filename}")
            
            return True
            
        except Exception as e:
            logger.error(f"处理small数据集失败: {str(e)}")
            return False
    
    def _process_100k_dataset(self, extract_path: Path, target_dir: Path) -> bool:
        """处理100k数据集"""
        try:
            # 100k数据集使用不同的文件格式
            # 处理评分数据 (u.data)
            ratings_file = extract_path / "u.data"
            if ratings_file.exists():
                # 读取评分数据 (user_id, item_id, rating, timestamp)
                ratings_df = pd.read_csv(ratings_file, sep='\t', 
                                       names=['userId', 'movieId', 'rating', 'timestamp'])
                ratings_df.to_csv(target_dir / "ratings.csv", index=False)
                logger.info(f"处理评分数据: {len(ratings_df)} 条评分")
            
            # 处理电影数据 (u.item)
            movies_file = extract_path / "u.item"
            if movies_file.exists():
                # 读取电影数据
                column_names = ['movieId', 'title', 'release_date', 'video_release_date', 'imdb_url'] + \
                              [f'genre_{i}' for i in range(19)]  # 19个genre列
                
                movies_df = pd.read_csv(movies_file, sep='|', encoding='latin-1', 
                                      names=column_names, header=None)
                
                # 处理genres - 将二进制编码转换为genre名称
                genre_names = ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                              'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                              'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
                
                def get_genres(row):
                    genres = []
                    for i, genre in enumerate(genre_names):
                        if row[f'genre_{i}'] == 1:
                            genres.append(genre)
                    return ', '.join(genres) if genres else 'unknown'
                
                movies_df['genres'] = movies_df.apply(get_genres, axis=1)
                
                # 只保留需要的列
                movies_final = movies_df[['movieId', 'title', 'genres']].copy()
                movies_final.to_csv(target_dir / "movies.csv", index=False)
                logger.info(f"处理电影数据: {len(movies_final)} 部电影")
            
            # 处理用户数据 (u.user)
            users_file = extract_path / "u.user"
            if users_file.exists():
                users_df = pd.read_csv(users_file, sep='|', 
                                     names=['userId', 'age', 'gender', 'occupation', 'zip_code'])
                users_df.to_csv(target_dir / "users.csv", index=False)
                logger.info(f"处理用户数据: {len(users_df)} 个用户")
            
            return True
            
        except Exception as e:
            logger.error(f"处理100k数据集失败: {str(e)}")
            return False
    
    def _process_1m_dataset(self, extract_path: Path, target_dir: Path) -> bool:
        """处理1m数据集"""
        try:
            # 处理评分数据 (ratings.dat)
            ratings_file = extract_path / "ratings.dat"
            if ratings_file.exists():
                ratings_df = pd.read_csv(ratings_file, sep='::', engine='python',
                                       names=['userId', 'movieId', 'rating', 'timestamp'])
                ratings_df.to_csv(target_dir / "ratings.csv", index=False)
                logger.info(f"处理评分数据: {len(ratings_df)} 条评分")
            
            # 处理电影数据 (movies.dat)
            movies_file = extract_path / "movies.dat"
            if movies_file.exists():
                movies_df = pd.read_csv(movies_file, sep='::', engine='python', encoding='latin-1',
                                      names=['movieId', 'title', 'genres'])
                
                # 处理genres
                movies_df['genres'] = movies_df['genres'].str.replace('|', ', ')
                movies_df.to_csv(target_dir / "movies.csv", index=False)
                logger.info(f"处理电影数据: {len(movies_df)} 部电影")
            
            # 处理用户数据 (users.dat)
            users_file = extract_path / "users.dat"
            if users_file.exists():
                users_df = pd.read_csv(users_file, sep='::', engine='python',
                                     names=['userId', 'gender', 'age', 'occupation', 'zip_code'])
                users_df.to_csv(target_dir / "users.csv", index=False)
                logger.info(f"处理用户数据: {len(users_df)} 个用户")
            
            return True
            
        except Exception as e:
            logger.error(f"处理1m数据集失败: {str(e)}")
            return False
    
    def _cleanup_temp_files(self):
        """清理临时文件"""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info("清理临时文件完成")
        except Exception as e:
            logger.warning(f"清理临时文件失败: {str(e)}")
    
    def generate_report(self, dataset_name: str = "small") -> Dict[str, Any]:
        """生成数据统计报告"""
        try:
            dataset_info = self.datasets[dataset_name]
            target_dir = self.base_dir / dataset_info["target_subdir"]
            
            if not target_dir.exists():
                logger.error(f"数据目录不存在: {target_dir}")
                return {}
            
            report = {
                "dataset_name": dataset_name,
                "data_directory": str(target_dir.absolute()),
                "timestamp": datetime.now().isoformat(),
                "files": {}
            }
            
            # 统计各文件
            for filename in ["movies.csv", "ratings.csv", "users.csv", "tags.csv", "links.csv"]:
                filepath = target_dir / filename
                if filepath.exists():
                    df = pd.read_csv(filepath)
                    report["files"][filename] = {
                        "rows": len(df),
                        "columns": list(df.columns),
                        "size_mb": filepath.stat().st_size / 1024 / 1024
                    }
            
            # 如果有评分数据，添加额外统计
            ratings_file = target_dir / "ratings.csv"
            if ratings_file.exists():
                ratings_df = pd.read_csv(ratings_file)
                report["statistics"] = {
                    "unique_users": ratings_df['userId'].nunique() if 'userId' in ratings_df.columns else 0,
                    "unique_movies": ratings_df['movieId'].nunique() if 'movieId' in ratings_df.columns else 0,
                    "total_ratings": len(ratings_df),
                    "rating_range": [float(ratings_df['rating'].min()), float(ratings_df['rating'].max())] if 'rating' in ratings_df.columns else [0, 0],
                    "avg_rating": float(ratings_df['rating'].mean()) if 'rating' in ratings_df.columns else 0
                }
            
            return report
            
        except Exception as e:
            logger.error(f"生成报告失败: {str(e)}")
            return {}

def main():
    """主函数"""
    downloader = MovieLensDataDownloader()
    
    print("MovieLens数据下载器 V2")
    print("=" * 50)
    
    # 选择数据集
    datasets = list(downloader.datasets.keys())
    print("\n可用数据集:")
    for i, dataset in enumerate(datasets, 1):
        info = downloader.datasets[dataset]
        print(f"{i}. {dataset}: {info['description']}")
    
    while True:
        try:
            choice = input(f"\n请选择数据集 (1-{len(datasets)}) [默认: 1]: ").strip()
            if not choice:
                choice = "1"
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(datasets):
                selected_dataset = datasets[choice_idx]
                break
            else:
                print("无效选择，请重新输入")
        except (ValueError, KeyboardInterrupt):
            print("\n用户取消操作")
            return
    
    # 下载数据
    force = input("\n是否强制重新下载? (y/N): ").strip().lower() == 'y'
    
    if downloader.download_dataset(selected_dataset, force_download=force):
        print("\n数据下载成功!")
        
        # 生成报告
        report = downloader.generate_report(selected_dataset)
        if report:
            print("\n数据统计报告:")
            print(f"数据集: {report['dataset_name']}")
            print(f"目录: {report['data_directory']}")
            print("\n文件统计:")
            for filename, stats in report["files"].items():
                print(f"  {filename}: {stats['rows']:,} 行, {stats['size_mb']:.2f} MB")
            
            if "statistics" in report:
                stats = report["statistics"]
                print(f"\n评分统计:")
                print(f"  用户数: {stats['unique_users']:,}")
                print(f"  电影数: {stats['unique_movies']:,}")
                print(f"  评分数: {stats['total_ratings']:,}")
                print(f"  评分范围: {stats['rating_range'][0]:.1f} - {stats['rating_range'][1]:.1f}")
                print(f"  平均评分: {stats['avg_rating']:.2f}")
        
    else:
        print("\n数据下载失败!")

if __name__ == "__main__":
    main()
