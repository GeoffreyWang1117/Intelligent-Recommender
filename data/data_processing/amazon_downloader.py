#!/usr/bin/env python3
"""
Amazon Reviews 2023 数据下载器
使用datasets库下载并保存为parquet格式

作者: GitHub Copilot
日期: 2025-08-27
"""

import pandas as pd
from pathlib import Path
import time

# 导入datasets库
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("❌ datasets库未安装，请运行: pip install datasets")

class AmazonDataDownloader:
    """Amazon Reviews 2023数据下载器"""
    
    def __init__(self):
        """初始化下载器"""
        # 项目根目录和数据目录
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / "data" / "amazon"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Amazon Reviews 2023主要类别
        self.main_categories = [
            "All_Beauty",
            "Amazon_Fashion", 
            "Arts_Crafts_and_Sewing",
            "Automotive",
            "Books",
            "CDs_and_Vinyl",
            "Cell_Phones_and_Accessories",
            "Clothing_Shoes_and_Jewelry",
            "Electronics",
            "Gift_Cards",
            "Grocery_and_Gourmet_Food",
            "Health_and_Household",
            "Home_and_Kitchen",
            "Industrial_and_Scientific",
            "Kindle_Store",
            "Movies_and_TV",
            "Musical_Instruments",
            "Office_Products",
            "Patio_Lawn_and_Garden",
            "Pet_Supplies",
            "Sports_and_Outdoors",
            "Tools_and_Home_Improvement",
            "Toys_and_Games",
            "Video_Games"
        ]
    
    def download_category(self, category: str) -> bool:
        """下载单个类别的数据
        
        Args:
            category: 类别名称
            
        Returns:
            bool: 下载是否成功
        """
        if not DATASETS_AVAILABLE:
            print("❌ datasets库不可用")
            return False
        
        print(f"\n📚 下载类别: {category}")
        print("-" * 40)
        
        # 检查文件是否已存在
        reviews_file = self.data_dir / f"{category}_reviews.parquet"
        meta_file = self.data_dir / f"{category}_meta.parquet"
        
        if reviews_file.exists() and meta_file.exists():
            print(f"✅ {category} 数据已存在，跳过下载")
            return True
        
        success_count = 0
        
        try:
            # 下载评论数据
            if not reviews_file.exists():
                print(f"🔽 下载 {category} 评论数据...")
                dataset_name = f"raw_review_{category}"
                
                dataset = load_dataset(
                    "McAuley-Lab/Amazon-Reviews-2023", 
                    dataset_name,
                    trust_remote_code=True
                )
                
                # 转换为DataFrame并保存
                df = dataset["full"].to_pandas()
                df.to_parquet(reviews_file, index=False)
                
                print(f"✅ 评论数据已保存: {len(df):,} 条 -> {reviews_file.name}")
                success_count += 1
            else:
                print(f"✅ 评论数据已存在: {reviews_file.name}")
                success_count += 1
            
        except Exception as e:
            print(f"❌ 评论数据下载失败: {e}")
        
        try:
            # 下载元数据
            if not meta_file.exists():
                print(f"🔽 下载 {category} 元数据...")
                dataset_name = f"raw_meta_{category}"
                
                dataset = load_dataset(
                    "McAuley-Lab/Amazon-Reviews-2023", 
                    dataset_name,
                    trust_remote_code=True
                )
                
                # 转换为DataFrame并保存
                df = dataset["full"].to_pandas()
                df.to_parquet(meta_file, index=False)
                
                print(f"✅ 元数据已保存: {len(df):,} 条 -> {meta_file.name}")
                success_count += 1
            else:
                print(f"✅ 元数据已存在: {meta_file.name}")
                success_count += 1
                
        except Exception as e:
            print(f"❌ 元数据下载失败: {e}")
        
        print(f"📊 {category} 完成: {success_count}/2 个文件")
        return success_count == 2
    
    def download_main_categories(self, categories=None):
        """下载主要类别的数据
        
        Args:
            categories: 要下载的类别列表，默认为前10个主要类别
        """
        if categories is None:
            categories = [
                "All_Beauty", "Books", "Electronics", "Movies_and_TV", 
                "Home_and_Kitchen", "Sports_and_Outdoors", "Toys_and_Games",
                "Automotive", "Arts_Crafts_and_Sewing", "Office_Products"
            ]
        
        print("🚀 开始下载Amazon Reviews 2023数据")
        print("=" * 50)
        print(f"📊 目标类别: {len(categories)} 个")
        print(f"💾 保存目录: {self.data_dir}")
        print("=" * 50)
        
        success_count = 0
        
        for i, category in enumerate(categories, 1):
            print(f"\n[{i}/{len(categories)}] 处理类别: {category}")
            
            if self.download_category(category):
                success_count += 1
            
            # 短暂休息
            time.sleep(2)
        
        print("\n" + "=" * 50)
        print("📊 下载总结")
        print("=" * 50)
        print(f"✅ 成功下载: {success_count}/{len(categories)} 个类别")
        print(f"💾 文件保存在: {self.data_dir}")
        
        # 显示文件列表
        self.list_downloaded_files()
    
    def download_all_categories(self):
        """下载所有支持的类别"""
        print("🚀 下载所有Amazon Reviews 2023类别")
        print("=" * 50)
        print(f"📊 总类别数: {len(self.main_categories)}")
        print("⚠️  这将下载大量数据，请确保有足够的存储空间")
        print("=" * 50)
        
        confirm = input("确认下载所有类别? (y/N): ").strip().lower()
        if confirm != 'y':
            print("❌ 取消下载")
            return
        
        self.download_main_categories(self.main_categories)
    
    def list_downloaded_files(self):
        """列出已下载的文件"""
        print("\n📋 已下载的文件:")
        print("-" * 40)
        
        parquet_files = list(self.data_dir.glob("*.parquet"))
        
        if not parquet_files:
            print("   ❌ 没有找到parquet文件")
            return
        
        # 按类别分组
        categories = set()
        for file in parquet_files:
            if "_reviews.parquet" in file.name:
                category = file.name.replace("_reviews.parquet", "")
                categories.add(category)
            elif "_meta.parquet" in file.name:
                category = file.name.replace("_meta.parquet", "")
                categories.add(category)
        
        total_size = 0
        for category in sorted(categories):
            review_file = self.data_dir / f"{category}_reviews.parquet"
            meta_file = self.data_dir / f"{category}_meta.parquet"
            
            review_size = review_file.stat().st_size if review_file.exists() else 0
            meta_size = meta_file.stat().st_size if meta_file.exists() else 0
            
            review_mb = review_size / (1024*1024)
            meta_mb = meta_size / (1024*1024)
            
            status = "✅" if review_file.exists() and meta_file.exists() else "⚠️ "
            
            print(f"   {status} {category}:")
            if review_file.exists():
                print(f"      📊 reviews: {review_mb:.1f} MB")
            if meta_file.exists():
                print(f"      🏷️  meta: {meta_mb:.1f} MB")
            
            total_size += review_size + meta_size
        
        print("-" * 40)
        print(f"📊 总大小: {total_size/(1024*1024):.1f} MB")
    
    def verify_files(self):
        """验证下载的文件"""
        print("\n🔍 验证Parquet文件:")
        print("-" * 30)
        
        parquet_files = list(self.data_dir.glob("*.parquet"))
        valid_count = 0
        
        for file in parquet_files:
            try:
                df = pd.read_parquet(file)
                print(f"✅ {file.name}: {len(df):,} 行, {len(df.columns)} 列")
                valid_count += 1
            except Exception as e:
                print(f"❌ {file.name}: 读取失败 - {e}")
        
        print("-" * 30)
        print(f"📊 验证结果: {valid_count}/{len(parquet_files)} 个文件有效")

def main():
    """主函数"""
    downloader = AmazonDataDownloader()
    
    print("Amazon Reviews 2023 数据下载器")
    print("1. 下载主要类别 (推荐)")
    print("2. 下载所有类别")
    print("3. 下载单个类别")
    print("4. 查看已下载文件")
    print("5. 验证文件")
    
    choice = input("\n请选择 (1-5): ").strip()
    
    if choice == "1":
        # 下载主要类别
        downloader.download_main_categories()
    
    elif choice == "2":
        # 下载所有类别
        downloader.download_all_categories()
    
    elif choice == "3":
        # 下载单个类别
        print(f"\n可选类别:")
        for i, cat in enumerate(downloader.main_categories[:15], 1):
            print(f"{i:2d}. {cat}")
        if len(downloader.main_categories) > 15:
            print(f"    ... 还有 {len(downloader.main_categories) - 15} 个类别")
        
        category = input("\n输入类别名称: ").strip()
        if category in downloader.main_categories:
            downloader.download_category(category)
        else:
            print("❌ 无效类别")
    
    elif choice == "4":
        # 查看已下载文件
        downloader.list_downloaded_files()
    
    elif choice == "5":
        # 验证文件
        downloader.verify_files()
    
    else:
        print("❌ 无效选择")

if __name__ == "__main__":
    main()
