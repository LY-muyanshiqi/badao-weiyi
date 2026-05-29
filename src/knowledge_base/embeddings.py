"""
水利大模型知识库 (RAG) — 规范检索 + 案例匹配 + 诊断报告生成
使用 ChromaDB + sentence-transformers 做向量检索
"""
import os
import json
import numpy as np
from pathlib import Path


class KnowledgeBase:
    """RAG knowledge base for hydraulic engineering standards and cases.

    Workflow:
      1. Chunk documents (standards, cases, reports)
      2. Embed with sentence-transformers
      3. Store in ChromaDB
      4. Retrieve relevant context for diagnosis explanation
    """

    def __init__(self, persist_dir='data/knowledge_base'):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection = None
        self.embedder = None
        self._initialized = False

    def initialize(self):
        """Lazy initialization of ChromaDB and embedding model."""
        if self._initialized:
            return

        try:
            import chromadb

            self.client = chromadb.PersistentClient(path=str(self.persist_dir))

            try:
                self.collection = self.client.get_collection("hydraulic_knowledge")
            except Exception:
                self.collection = self.client.create_collection(
                    name="hydraulic_knowledge",
                    metadata={"description": "水利工程安全诊断知识库"}
                )

        except ImportError:
            print("ChromaDB not installed. Run: pip install chromadb")
            self.client = None
            self.collection = None

        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        except ImportError:
            print("sentence-transformers not installed. Run: pip install sentence-transformers")
            self.embedder = None

        self._initialized = True

    def add_documents(self, documents, metadatas=None, ids=None):
        """Add documents to the knowledge base.

        Args:
            documents: list of text strings
            metadatas: list of dicts with metadata
            ids: list of document IDs
        """
        if self.collection is None or self.embedder is None:
            self.initialize()
        if self.collection is None or self.embedder is None:
            return

        embeddings = self.embedder.encode(documents, show_progress_bar=False).tolist()

        if ids is None:
            existing_count = self.collection.count()
            ids = [f"doc_{existing_count + i}" for i in range(len(documents))]

        if metadatas is None:
            metadatas = [{} for _ in documents]

        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

    def search(self, query, n_results=5):
        """Search for relevant knowledge given a query.

        Args:
            query: natural language query string
            n_results: number of results to return

        Returns:
            list of dicts with 'content', 'metadata', 'distance'
        """
        if self.collection is None or self.embedder is None:
            self.initialize()
        if self.collection is None or self.embedder is None:
            return self._fallback_search(query, n_results)

        query_embedding = self.embedder.encode([query], show_progress_bar=False).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
        )

        return [
            {
                'content': doc,
                'metadata': meta,
                'distance': dist,
            }
            for doc, meta, dist in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )
        ]

    def _fallback_search(self, query, n_results=5):
        """Fallback keyword-based search when ChromaDB unavailable."""
        return [{
            'content': f'[模拟检索结果] 与查询 "{query}" 相关的水利规范条目',
            'metadata': {'source': 'fallback', 'type': 'standard'},
            'distance': 0.5,
        }]


class StandardKnowledgeBase(KnowledgeBase):
    """Pre-loaded knowledge base with Chinese hydraulic engineering standards."""

    STANDARDS = [
        {
            'id': 'SL252-2017',
            'title': '水利水电工程等级划分及洪水标准',
            'excerpts': [
                '水利水电工程的等别根据其工程规模、效益及在国民经济中的重要性划分为Ⅰ、Ⅱ、Ⅲ、Ⅳ、Ⅴ五等。',
                '大坝安全监测应根据工程等别、坝型、坝高、地质条件等因素确定监测项目与频次。',
                '渗透压力监测是土石坝安全监测的核心内容之一。',
            ]
        },
        {
            'id': 'SL601-2019',
            'title': '混凝土坝安全监测技术规范',
            'excerpts': [
                '变形监测基准点应设在变形影响范围以外的稳定区域。',
                '渗流监测应包括渗流量、扬压力、绕坝渗流等内容。',
                '环境量监测应包括库水位、气温、降雨量、水温等。',
            ]
        },
        {
            'id': 'GB50487-2008',
            'title': '水利水电工程地质勘察规范',
            'excerpts': [
                '对于高地震烈度区，应进行专门的工程场地地震安全性评价。',
                '坝基岩体质量分级应考虑岩石强度、完整性、结构面特征等因素。',
                '土石坝坝基应重点查明渗透稳定性问题。',
            ]
        },
    ]

    def load_standards(self):
        """Load pre-defined hydraulic engineering standards into the KB."""
        self.initialize()
        docs = []
        metas = []
        ids = []

        for std in self.STANDARDS:
            for i, excerpt in enumerate(std['excerpts']):
                docs.append(excerpt)
                metas.append({
                    'standard_id': std['id'],
                    'title': std['title'],
                    'type': 'standard',
                })
                ids.append(f"{std['id']}_{i}")

        self.add_documents(docs, metas, ids)
        print(f"Loaded {len(docs)} standard excerpts from {len(self.STANDARDS)} documents")
        return len(docs)


def generate_diagnosis_report(diagnosis_result, kb=None):
    """Generate a human-readable diagnosis report with RAG context.

    Args:
        diagnosis_result: output from DiagnosticEngine.diagnose()
        kb: KnowledgeBase instance (optional)

    Returns:
        dict with 'summary', 'details', 'recommendations', 'references'
    """
    risk = diagnosis_result['overall_risk_level']
    dominant_hazard = diagnosis_result['dominant_hazard']
    per_hazard = diagnosis_result['per_hazard_scores']

    # Generate summary
    summaries = {
        '安全': '经综合评估，工程结构处于安全状态，各监测指标均在正常范围内。',
        '注意': f'工程整体处于可控状态，但{per_hazard[dominant_hazard]["name"]}指标出现轻微异常，建议加强监测。',
        '警告': f'工程存在中等风险，{per_hazard[dominant_hazard]["name"]}指标超限，建议尽快安排现场检查和维修。',
        '危险': f'工程处于高风险状态，{per_hazard[dominant_hazard]["name"]}指标严重超限，应立即启动应急预案，组织专家会诊。',
    }

    # Generate recommendations
    recommendations = []
    if risk == '安全':
        recommendations = [
            '继续执行常规监测计划，监测频次可维持现状。',
            '定期校准传感器设备，确保数据质量。',
        ]
    elif risk == '注意':
        recommendations = [
            f'加密{per_hazard[dominant_hazard]["name"]}相关监测点位的采集频次至每日1次。',
            '对异常区域进行现场勘查，排除传感器故障可能。',
            '准备应急预案，确保人员物资到位。',
        ]
    elif risk == '警告':
        recommendations = [
            f'立即对{per_hazard[dominant_hazard]["name"]}影响区域进行详细检查。',
            '将监测频次提升至实时连续监测。',
            '组织专家会诊，制定修复方案。',
            '通知下游相关单位做好应急准备。',
        ]
    else:  # 危险
        recommendations = [
            '立即启动应急预案，组织受影响区域人员撤离。',
            '通知水利主管部门和相关防汛指挥机构。',
            '连续监测关键指标变化趋势，每分钟汇报一次。',
            '暂停工程运行，准备抢险物资和队伍。',
        ]

    # Retrieve relevant standards
    references = []
    if kb:
        try:
            query = f"{dominant_hazard} {risk} 诊断"
            results = kb.search(query, n_results=3)
            references = [
                {'source': r['metadata'].get('title', r['metadata'].get('standard_id', '')),
                 'content': r['content'][:200],
                 'relevance': f"{1 - r['distance']:.1%}"}
                for r in results
            ]
        except Exception:
            references = [{'source': '离线模式', 'content': '知识库未加载，无法提供规范引用'}]

    return {
        'summary': summaries.get(risk, summaries['注意']),
        'overall_risk': risk,
        'dominant_hazard': per_hazard[dominant_hazard]['name'],
        'hazard_details': per_hazard,
        'recommendations': recommendations,
        'references': references,
        'damage_span': diagnosis_result.get('mean_damage_span', 0),
    }


if __name__ == '__main__':
    kb = StandardKnowledgeBase()
    kb.load_standards()

    # Test search
    results = kb.search('渗流 稳定性 监测')
    for r in results:
        print(f"[{r['metadata'].get('standard_id', '?')}] {r['content'][:100]}... (dist={r['distance']:.3f})")
