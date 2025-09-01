"""
多Teacher融合模块

提供多种Teacher融合策略和算法
支持Fisher引导的智能融合
"""

from .multi_teacher_fusion import MultiTeacherFusion

__all__ = [
    "MultiTeacherFusion"
]
