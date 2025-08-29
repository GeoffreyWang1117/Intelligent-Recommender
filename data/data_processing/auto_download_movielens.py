#!/usr/bin/env python3
"""
自动下载MovieLens数据 V2
非交互式脚本，自动下载所有三个数据集到对应子目录

数据存储结构:
- data/movielens/small/     (MovieLens Latest Small)
- data/movielens/100k/      (MovieLens 100K)  
- data/movielens/1m/        (MovieLens 1M)

作者: GitHub Copilot
日期: 2025-08-27
"""

import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent))

from download_real_movielens import MovieLensDataDownloader
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """自动下载所有数据集"""
    downloader = MovieLensDataDownloader()
    
    print("自动下载MovieLens数据集到分类子目录")
    print("=" * 50)
    
    # 要下载的数据集列表
    datasets_to_download = ["small", "100k", "1m"]
    
    for dataset_name in datasets_to_download:
        print(f"\n开始处理数据集: {dataset_name}")
        print("-" * 30)
        
        success = downloader.download_dataset(dataset_name, force_download=False)
        
        if success:
            print(f"✓ {dataset_name} 数据集下载成功")
            
            # 生成报告
            report = downloader.generate_report(dataset_name)
            if report:
                print(f"  存储位置: {report['data_directory']}")
                if "statistics" in report:
                    stats = report["statistics"]
                    print(f"  用户数: {stats['unique_users']:,}")
                    print(f"  电影数: {stats['unique_movies']:,}")
                    print(f"  评分数: {stats['total_ratings']:,}")
        else:
            print(f"✗ {dataset_name} 数据集下载失败")
    
    print("\n" + "=" * 50)
    print("所有数据集处理完成!")
    
    # 显示最终目录结构
    base_dir = Path("data/movielens")
    if base_dir.exists():
        print("\n最终目录结构:")
        for subdir in sorted(base_dir.iterdir()):
            if subdir.is_dir():
                files = list(subdir.glob("*.csv"))
                print(f"  {subdir.name}/")
                for file in sorted(files):
                    size_mb = file.stat().st_size / 1024 / 1024
                    print(f"    {file.name} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
