"""中文说明：CEDG 第一阶段建模包，提供数据张量化工具和 PyTorch scorer/ranker。

CEDG-Set modeling package.

This package contains the first functional PyTorch implementation of the
CEDG-Score MVP: parent peptide context plus an edit set predicts delta PAMPA.
It intentionally keeps chemistry payloads as structured categorical metadata
for now, while preserving the module boundary where an atom-level payload
encoder can be added later.
"""
