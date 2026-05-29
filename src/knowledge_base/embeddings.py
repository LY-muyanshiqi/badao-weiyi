"""
水利大模型知识库 (RAG) — 规范检索 + 案例匹配 + 诊断报告生成
向量检索: Modelscope (国内) > TF-IDF (离线回退) > ChromaDB
"""
import os
import re
import json
import numpy as np
from pathlib import Path
from collections import Counter


class TfidfEmbedder:
    """离线 TF-IDF 嵌入器 — 不依赖任何外部模型下载，纯本地计算"""

    def __init__(self):
        self.vocabulary = {}
        self.idf = {}
        self.doc_count = 0

    def _tokenize(self, text):
        """Simple Chinese + word tokenizer using character bigrams."""
        # Extract Chinese characters and alphanumeric tokens
        tokens = []
        # Chinese bigrams
        chinese = re.findall(r'[一-鿿]+', text)
        for seg in chinese:
            for i in range(len(seg) - 1):
                tokens.append(seg[i:i+2])
            tokens.append(seg[-1])  # unigram for odd char
        # Alphanumeric
        alpha = re.findall(r'[a-zA-Z0-9]+', text)
        tokens.extend(alpha)
        return tokens

    def fit(self, documents):
        """Build vocabulary and IDF from document corpus."""
        self.doc_count = len(documents)
        df = Counter()

        for doc in documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                df[token] += 1

        self.idf = {t: np.log((self.doc_count + 1) / (df[t] + 1)) + 1 for t in df}
        self.vocabulary = {t: i for i, t in enumerate(sorted(self.idf.keys()))}

    def encode(self, texts, show_progress_bar=False):
        """Encode texts as TF-IDF vectors."""
        vectors = np.zeros((len(texts), len(self.vocabulary)))
        for i, text in enumerate(texts):
            tokens = self._tokenize(text)
            tf = Counter(tokens)
            total = sum(tf.values()) + 1
            for token, count in tf.items():
                if token in self.vocabulary:
                    j = self.vocabulary[token]
                    vectors[i, j] = (count / total) * self.idf.get(token, 1.0)
            # L2 normalize
            norm = np.linalg.norm(vectors[i]) + 1e-10
            vectors[i] /= norm
        return vectors.tolist()


class KnowledgeBase:
    """RAG knowledge base — Modelscope embedding with TF-IDF fallback."""

    def __init__(self, persist_dir='data/knowledge_base'):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection = None
        self.embedder = None
        self._initialized = False
        self._embedder_source = 'none'

    def _init_modelscope_embedder(self):
        """Try loading embedding model from Modelscope (accessible in China)."""
        try:
            from modelscope.models import Model
            from modelscope.pipelines import pipeline

            model_id = 'iic/nlp_corom_sentence-embedding_chinese-base'
            self.embedder = Model.from_pretrained(model_id)
            self._embedder_source = 'modelscope'
            print(f"Embedder: Modelscope/{model_id}")
            return True
        except Exception:
            return False

    def _init_transformers_embedder(self):
        """Try loading from HuggingFace (may be blocked in China)."""
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(
                'paraphrase-multilingual-MiniLM-L12-v2',
                device='cpu',
            )
            self._embedder_source = 'sentence-transformers'
            print("Embedder: sentence-transformers (HuggingFace)")
            return True
        except Exception:
            return False

    def _init_tfidf_embedder(self):
        """Offline TF-IDF fallback — always works."""
        self.embedder = TfidfEmbedder()
        self._embedder_source = 'tfidf'
        print("Embedder: TF-IDF (offline)")
        return True

    def initialize(self):
        """Lazy init — TF-IDF offline (always works, no network needed).

        For ChromaDB + neural embeddings, install and configure:
          pip install chromadb sentence-transformers
        """
        if self._initialized:
            return

        # Init ChromaDB if available
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
            self.client = None
            self.collection = None

        # TF-IDF offline embedder — zero network, always works
        self._init_tfidf_embedder()

        self._initialized = True

    def add_documents(self, documents, metadatas=None, ids=None):
        if self.embedder is None:
            self.initialize()

        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        if metadatas is None:
            metadatas = [{} for _ in documents]

        # TF-IDF mode: store locally, refit on all docs
        if self._embedder_source == 'tfidf':
            if not hasattr(self, '_tfidf_docs'):
                self._tfidf_docs = []
                self._tfidf_metas = []
                self._tfidf_ids = []
            self._tfidf_docs.extend(documents)
            self._tfidf_metas.extend(metadatas)
            self._tfidf_ids.extend(ids)
            self.embedder.fit(self._tfidf_docs)
            return

        # ChromaDB mode
        if self.collection is None:
            return
        embeddings = self.embedder.encode(documents, show_progress_bar=False)
        if isinstance(embeddings, list) and len(embeddings) > 0 and isinstance(embeddings[0], list):
            pass  # already list of lists
        else:
            embeddings = embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings

        existing_count = self.collection.count()
        ids = [f"doc_{existing_count + i}" for i in range(len(documents))]

        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

    def search(self, query, n_results=5):
        if self.embedder is None:
            self.initialize()
        if self.embedder is None:
            return self._fallback_search(query, n_results)

        # TF-IDF mode: compute similarity manually
        if self._embedder_source == 'tfidf':
            if not hasattr(self, '_tfidf_docs') or not self._tfidf_docs:
                return self._fallback_search(query, n_results)
            q_vec = np.array(self.embedder.encode([query])[0])
            doc_vecs = np.array(self.embedder.encode(self._tfidf_docs))
            # Cosine similarity
            q_norm = np.linalg.norm(q_vec) + 1e-10
            d_norms = np.linalg.norm(doc_vecs, axis=1) + 1e-10
            sims = np.dot(doc_vecs, q_vec) / (d_norms * q_norm)
            top_k = np.argsort(sims)[::-1][:n_results]
            return [
                {
                    'content': self._tfidf_docs[i],
                    'metadata': self._tfidf_metas[i],
                    'distance': float(1 - sims[i]),
                }
                for i in top_k if sims[i] > 0.01
            ]

        # ChromaDB mode
        if self.collection is None:
            return self._fallback_search(query, n_results)

        q_vec = self.embedder.encode([query], show_progress_bar=False)
        if hasattr(q_vec, 'tolist'):
            q_vec = q_vec.tolist()

        results = self.collection.query(
            query_embeddings=q_vec,
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
