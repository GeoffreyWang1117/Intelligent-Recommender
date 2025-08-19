import numpy as np
import faiss
import pickle
import os
import logging

logger = logging.getLogger(__name__)

class FAISSIndex:
    """FAISS向量索引，用于快速相似度搜索"""
    
    def __init__(self, dimension, index_type='IVFFlat', nlist=100):
        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist
        self.index = None
        self.item_ids = None
        self.item_vectors = None
        
    def build_index(self, item_vectors, item_ids):
        """构建FAISS索引"""
        try:
            self.item_vectors = np.array(item_vectors).astype('float32')
            self.item_ids = np.array(item_ids)
            
            if len(self.item_vectors.shape) != 2:
                raise ValueError("向量必须是二维数组")
            
            if self.item_vectors.shape[1] != self.dimension:
                raise ValueError(f"向量维度不匹配，期望{self.dimension}，实际{self.item_vectors.shape[1]}")
            
            # 创建索引
            if self.index_type == 'IVFFlat':
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
            elif self.index_type == 'Flat':
                self.index = faiss.IndexFlatL2(self.dimension)
            else:
                raise ValueError(f"不支持的索引类型: {self.index_type}")
            
            # 训练索引（对于IVF索引需要）
            if self.index_type == 'IVFFlat':
                self.index.train(self.item_vectors)
            
            # 添加向量到索引
            self.index.add(self.item_vectors)
            
            logger.info(f"FAISS索引构建完成，共{len(item_ids)}个向量")
            
        except Exception as e:
            logger.error(f"FAISS索引构建失败: {e}")
            raise
    
    def search(self, query_vector, top_k=10):
        """搜索最相似的向量"""
        try:
            if self.index is None:
                raise ValueError("索引未构建")
            
            query_vector = np.array(query_vector).astype('float32')
            if len(query_vector.shape) == 1:
                query_vector = query_vector.reshape(1, -1)
            
            # 搜索
            distances, indices = self.index.search(query_vector, top_k)
            
            # 返回结果
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx >= 0:  # 有效索引
                    similarity = 1.0 / (1.0 + dist)  # 转换为相似度
                    results.append({
                        'item_id': self.item_ids[idx],
                        'similarity': float(similarity),
                        'distance': float(dist)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"FAISS搜索失败: {e}")
            return []
    
    def batch_search(self, query_vectors, top_k=10):
        """批量搜索"""
        try:
            if self.index is None:
                raise ValueError("索引未构建")
            
            query_vectors = np.array(query_vectors).astype('float32')
            
            # 搜索
            distances, indices = self.index.search(query_vectors, top_k)
            
            # 返回结果
            batch_results = []
            for i in range(len(query_vectors)):
                results = []
                for dist, idx in zip(distances[i], indices[i]):
                    if idx >= 0:
                        similarity = 1.0 / (1.0 + dist)
                        results.append({
                            'item_id': self.item_ids[idx],
                            'similarity': float(similarity),
                            'distance': float(dist)
                        })
                batch_results.append(results)
            
            return batch_results
            
        except Exception as e:
            logger.error(f"FAISS批量搜索失败: {e}")
            return []
    
    def add_vectors(self, new_vectors, new_item_ids):
        """添加新向量到索引"""
        try:
            if self.index is None:
                raise ValueError("索引未构建")
            
            new_vectors = np.array(new_vectors).astype('float32')
            new_item_ids = np.array(new_item_ids)
            
            # 添加到索引
            self.index.add(new_vectors)
            
            # 更新item_ids和vectors
            self.item_vectors = np.vstack([self.item_vectors, new_vectors])
            self.item_ids = np.concatenate([self.item_ids, new_item_ids])
            
            logger.info(f"添加了{len(new_item_ids)}个新向量到索引")
            
        except Exception as e:
            logger.error(f"添加向量失败: {e}")
            raise
    
    def get_vector_by_id(self, item_id):
        """根据item_id获取向量"""
        try:
            if self.item_ids is None:
                return None
            
            mask = self.item_ids == item_id
            if not np.any(mask):
                return None
            
            index = np.where(mask)[0][0]
            return self.item_vectors[index]
            
        except Exception as e:
            logger.error(f"获取向量失败: {e}")
            return None
    
    def save_index(self, index_path):
        """保存索引"""
        try:
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            
            # 保存FAISS索引
            faiss.write_index(self.index, index_path)
            
            # 保存元数据
            metadata_path = index_path.replace('.bin', '_metadata.pkl')
            metadata = {
                'item_ids': self.item_ids,
                'item_vectors': self.item_vectors,
                'dimension': self.dimension,
                'index_type': self.index_type,
                'nlist': self.nlist
            }
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"FAISS索引已保存到: {index_path}")
            
        except Exception as e:
            logger.error(f"FAISS索引保存失败: {e}")
            raise
    
    def load_index(self, index_path):
        """加载索引"""
        try:
            # 加载FAISS索引
            self.index = faiss.read_index(index_path)
            
            # 加载元数据
            metadata_path = index_path.replace('.bin', '_metadata.pkl')
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            self.item_ids = metadata['item_ids']
            self.item_vectors = metadata['item_vectors']
            self.dimension = metadata['dimension']
            self.index_type = metadata['index_type']
            self.nlist = metadata['nlist']
            
            logger.info(f"FAISS索引已从 {index_path} 加载")
            
        except Exception as e:
            logger.error(f"FAISS索引加载失败: {e}")
            raise
    
    def get_index_info(self):
        """获取索引信息"""
        if self.index is None:
            return None
        
        return {
            'dimension': self.dimension,
            'index_type': self.index_type,
            'total_vectors': self.index.ntotal,
            'is_trained': self.index.is_trained if hasattr(self.index, 'is_trained') else True
        }
