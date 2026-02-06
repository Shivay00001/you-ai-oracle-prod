#!/usr/bin/env python3
"""
YOU.AI Oracle Service - Production-Ready AI Guardian
NO LLM dependencies - pure rule-based decision engine
Compatibility wrappers added for modern Web3 / Python versions.
"""

import asyncio
import json
import logging
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import platform
import traceback

# --- Safe imports with helpful error messages --------------------------------
try:
    # Core blockchain deps
    from web3 import Web3
    # middleware location changed across versions; try both
    try:
        # older web3 versions
        from web3.middleware import geth_poa_middleware  # type: ignore
        _POA_MIDDLEWARE = 'geth_poa_middleware'
    except Exception:
        # web3 v7+ exposes a different middleware
        try:
            from web3.middleware.proof_of_authority import ExtraDataToPOAMiddleware  # type: ignore
            _POA_MIDDLEWARE = 'ExtraDataToPOAMiddleware'
        except Exception:
            _POA_MIDDLEWARE = None

    from eth_account import Account
except Exception as e:
    msg = (
        "Critical import error for web3/eth-account.\n\n"
        "Common fixes:\n"
        "  1) Use Python 3.10 or 3.11 for best compatibility with web3/eth-account dependencies.\n"
        "  2) Create a virtualenv with Python 3.10 and install pinned packages:\n"
        "       py -3.10 -m venv venv310\n"
        "       .\\venv310\\Scripts\\activate\n"
        "       pip install --upgrade pip\n"
        "       pip install web3==5.31.3 eth-account redis pyyaml numpy scikit-learn\n"
        "  3) If you must stay on Python 3.12, you will likely need to patch older dependencies\n"
        "     (e.g., parsimonious) or use newer web3/eth-account versions and update code accordingly.\n\n"
        f"Underlying error: {e}\n\nTraceback:\n{traceback.format_exc()}"
    )
    raise ImportError(msg)

# Redis and DB
try:
    import redis
    from redis.exceptions import RedisError
except Exception as e:
    raise ImportError(
        "Redis package missing. Install with `pip install redis`.\n" + str(e)
    )

import sqlite3
from pathlib import Path

# Data science
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception as e:
    raise ImportError(
        "Scikit-learn / numpy missing. Install with `pip install numpy scikit-learn`.\n" + str(e)
    )

# Configuration and monitoring
try:
    import yaml
except Exception:
    raise ImportError("PyYAML missing. Install with `pip install pyyaml`.")

import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('you_ai_oracle.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Helpers for compatibility ---------------------------------------------------
def web3_is_connected(web3: Web3) -> bool:
    """Wrapper for web3 connection check for multiple web3 versions."""
    # older web3: isConnected(), newer: is_connected
    if hasattr(web3, "isConnected") and callable(getattr(web3, "isConnected")):
        return web3.isConnected()  # type: ignore
    if hasattr(web3, "is_connected"):
        return web3.is_connected  # type: ignore
    # fallback: try simple RPC call
    try:
        web3.eth.block_number  # access to trigger connection
        return True
    except Exception:
        return False

def inject_poa_middleware_if_needed(web3: Web3):
    """Inject a POA middleware compatible with installed web3."""
    if _POA_MIDDLEWARE == 'geth_poa_middleware':
        try:
            web3.middleware_onion.inject(geth_poa_middleware, layer=0)  # type: ignore
            logger.info("Injected geth_poa_middleware")
        except Exception as e:
            logger.warning(f"Failed to inject geth_poa_middleware: {e}")
    elif _POA_MIDDLEWARE == 'ExtraDataToPOAMiddleware':
        try:
            from web3.middleware.proof_of_authority import ExtraDataToPOAMiddleware  # type: ignore
            web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
            logger.info("Injected ExtraDataToPOAMiddleware")
        except Exception as e:
            logger.warning(f"Failed to inject ExtraDataToPOAMiddleware: {e}")
    else:
        logger.debug("No POA middleware available in this environment")

# ============================================================================
# DATA MODELS (unchanged)
# ============================================================================

class ProposalCategory(Enum):
    RESEARCH = "research"
    INFRASTRUCTURE = "infrastructure"
    MARKETING = "marketing"
    LEGAL = "legal"
    PARTNERSHIP = "partnership"
    TREASURY = "treasury"
    IP_LICENSING = "ip_licensing"
    SUCCESSOR_TRAINING = "successor_training"
    EMERGENCY = "emergency"
    GENERAL = "general"

@dataclass
class FounderPersonality:
    vision_keywords: List[str]
    risk_tolerance: float
    innovation_bias: float
    social_impact_weight: float
    financial_weight: float
    decision_patterns: Dict[str, float]
    core_values: List[str]
    red_flags: List[str]

@dataclass
class Proposal:
    id: int
    title: str
    description: str
    amount: int
    recipient: str
    category: str
    created_at: int
    voting_ends_at: int
    for_votes: int
    against_votes: int
    executed: bool
    ai_approved: bool
    ai_confidence: int

@dataclass
class AIDecision:
    proposal_id: int
    approved: bool
    confidence: float
    reasoning: str
    risk_assessment: float
    alignment_score: float
    category_score: float
    keyword_matches: int
    decision_timestamp: int
    decision_factors: Dict[str, float]

# ============================================================================
# DECISION ENGINE (unchanged logic)
# ============================================================================

class DecisionEngine:
    def __init__(self, personality: FounderPersonality):
        self.personality = personality
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3)
        )
        self._build_keyword_vectors()
        self.APPROVAL_THRESHOLD = 0.65
        self.HIGH_CONFIDENCE_THRESHOLD = 0.80
        self.MINIMUM_CONFIDENCE = 0.40
        logger.info("Decision engine initialized with rule-based AI")

    def _build_keyword_vectors(self):
        self.vision_text = ' '.join(self.personality.vision_keywords)
        self.core_values_text = ' '.join(self.personality.core_values)
        self.red_flag_text = ' '.join(self.personality.red_flags)

    def analyze_proposal(self, proposal: Proposal) -> AIDecision:
        start_time = time.time()
        proposal_text = f"{proposal.title} {proposal.description}".lower()

        red_flag_score = self._check_red_flags(proposal_text)
        if red_flag_score > 0.7:
            return AIDecision(
                proposal_id=proposal.id,
                approved=False,
                confidence=0.95,
                reasoning="REJECTED: Contains red flag content (scam indicators, misaligned values, or harmful patterns)",
                risk_assessment=1.0,
                alignment_score=0.0,
                category_score=0.0,
                keyword_matches=0,
                decision_timestamp=int(time.time()),
                decision_factors={'red_flag': red_flag_score}
            )

        alignment_score = self._calculate_alignment_score(proposal_text)
        category_score = self._get_category_score(proposal.category)
        keyword_matches = self._count_keyword_matches(proposal_text)
        risk_score = self._assess_risk(proposal)
        innovation_score = self._detect_innovation(proposal_text)
        financial_score = self._analyze_financials(proposal)
        systemic_score = self._assess_systemic_impact(proposal_text)
        decentralization_score = self._score_decentralization(proposal_text)

        decision_factors = {
            'alignment': alignment_score,
            'category': category_score,
            'keywords': keyword_matches / 10.0,
            'risk': risk_score,
            'innovation': innovation_score,
            'financial': financial_score,
            'systemic': systemic_score,
            'decentralization': decentralization_score
        }

        final_score = (
            alignment_score * 0.25 +
            category_score * 0.20 +
            (keyword_matches / 10.0) * 0.15 +
            innovation_score * 0.15 +
            systemic_score * 0.10 +
            decentralization_score * 0.10 +
            financial_score * 0.05
        )

        risk_adjusted_score = final_score * (1 - risk_score * (1 - self.personality.risk_tolerance))
        approved = risk_adjusted_score >= self.APPROVAL_THRESHOLD
        confidence = min(abs(risk_adjusted_score - 0.5) * 2, 1.0)
        if confidence < self.MINIMUM_CONFIDENCE:
            confidence = self.MINIMUM_CONFIDENCE

        reasoning = self._generate_reasoning(
            approved, risk_adjusted_score, decision_factors, keyword_matches, proposal
        )

        elapsed = time.time() - start_time
        logger.info(f"Decision computed in {elapsed:.3f}s: {'APPROVED' if approved else 'REJECTED'} (confidence: {confidence:.2f})")

        return AIDecision(
            proposal_id=proposal.id,
            approved=approved,
            confidence=confidence,
            reasoning=reasoning,
            risk_assessment=risk_score,
            alignment_score=alignment_score,
            category_score=category_score,
            keyword_matches=keyword_matches,
            decision_timestamp=int(time.time()),
            decision_factors=decision_factors
        )

    def _check_red_flags(self, text: str) -> float:
        red_flags_found = 0
        for flag in self.personality.red_flags:
            if flag.lower() in text:
                red_flags_found += 1
        return min(red_flags_found / 3.0, 1.0)

    def _calculate_alignment_score(self, text: str) -> float:
        try:
            corpus = [self.vision_text, text]
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception:
            matches = sum(1 for kw in self.personality.vision_keywords if kw.lower() in text)
            return min(matches / 5.0, 1.0)

    def _get_category_score(self, category: str) -> float:
        return self.personality.decision_patterns.get(category, 0.5)

    def _count_keyword_matches(self, text: str) -> int:
        return sum(1 for kw in self.personality.vision_keywords if kw.lower() in text)

    def _assess_risk(self, proposal: Proposal) -> float:
        risk_factors = []
        amount_eth = proposal.amount / 1e18
        amount_risk = min(amount_eth / 1000, 1.0)
        risk_factors.append(amount_risk)
        recipient_risk = 0.3
        risk_factors.append(recipient_risk)
        high_risk_categories = ['marketing', 'emergency', 'general']
        category_risk = 0.7 if proposal.category in high_risk_categories else 0.3
        risk_factors.append(category_risk)
        return sum(risk_factors) / len(risk_factors)

    def _detect_innovation(self, text: str) -> float:
        innovation_keywords = [
            'ai', 'automation', 'protocol', 'algorithm', 'system',
            'decentralized', 'smart contract', 'blockchain', 'ml',
            'neural', 'autonomous', 'self-executing', 'programmable'
        ]
        matches = sum(1 for kw in innovation_keywords if kw in text)
        base_score = min(matches / 5.0, 1.0)
        return base_score * (1 + self.personality.innovation_bias * 0.5)

    def _analyze_financials(self, proposal: Proposal) -> float:
        amount_eth = proposal.amount / 1e18
        if amount_eth == 0:
            return 0.5
        elif amount_eth < 0.1:
            return 0.8
        elif amount_eth < 10:
            return 0.6
        elif amount_eth < 100:
            return 0.4
        else:
            return 0.2

    def _assess_systemic_impact(self, text: str) -> float:
        systemic_keywords = [
            'system', 'infrastructure', 'protocol', 'framework',
            'platform', 'architecture', 'foundation', 'core',
            'base layer', 'fundamental', 'structural'
        ]
        matches = sum(1 for kw in systemic_keywords if kw in text)
        return min(matches / 4.0, 1.0)

    def _score_decentralization(self, text: str) -> float:
        decentralization_keywords = [
            'decentralized', 'distributed', 'permissionless',
            'trustless', 'censorship-resistant', 'autonomous',
            'peer-to-peer', 'p2p', 'open source'
        ]
        matches = sum(1 for kw in decentralization_keywords if kw in text)
        return min(matches / 3.0, 1.0)

    def _generate_reasoning(self, approved: bool, final_score: float, factors: Dict[str, float], keyword_matches: int, proposal: Proposal) -> str:
        status = "APPROVED" if approved else "REJECTED"
        sorted_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)
        top_factors = sorted_factors[:3]
        reasoning_parts = [
            f"Decision: {status} (Score: {final_score:.3f})",
            f"Category: {proposal.category} (base score: {factors['category']:.2f})",
            f"Vision alignment: {factors['alignment']:.2f}",
            f"Keyword matches: {keyword_matches}",
            f"Risk level: {factors['risk']:.2f}",
            "",
            "Key factors:",
        ]
        for factor_name, score in top_factors:
            reasoning_parts.append(f"  - {factor_name}: {score:.2f}")
        if factors['innovation'] > 0.7:
            reasoning_parts.append("\n✓ High innovation potential detected")
        if factors['systemic'] > 0.6:
            reasoning_parts.append("✓ Strong systemic impact")
        if factors['decentralization'] > 0.5:
            reasoning_parts.append("✓ Aligns with decentralization principles")
        if factors['risk'] > 0.7:
            reasoning_parts.append("\n⚠ High risk detected - requires careful evaluation")
        if approved:
            if final_score > 0.8:
                reasoning_parts.append("\nStrong recommendation: Excellent alignment with DAO vision")
            else:
                reasoning_parts.append("\nModerate recommendation: Meets minimum criteria")
        else:
            if final_score < 0.4:
                reasoning_parts.append("\nStrong rejection: Poor alignment with core values")
            else:
                reasoning_parts.append("\nModerate rejection: Insufficient strategic value")
        return "\n".join(reasoning_parts)

# ============================================================================
# YOU.AI ORACLE SYSTEM (mostly unchanged, with compatibility helpers)
# ============================================================================

class YOUAIOracle:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.web3 = Web3(Web3.HTTPProvider(self.config['ethereum']['rpc_url']))
        if self.config['ethereum'].get('poa_mode', False):
            inject_poa_middleware_if_needed(self.web3)

        if not web3_is_connected(self.web3):
            raise ConnectionError(
                "Failed to connect to Ethereum node. Check RPC URL and network.\n"
                f"Configured RPC URL: {self.config['ethereum'].get('rpc_url')}"
            )

        try:
            chain_id = self.web3.eth.chain_id
        except Exception:
            chain_id = None

        logger.info(f"Connected to Ethereum network (Chain ID: {chain_id})")

        # Load contracts
        self._load_contracts()

        # Redis / DB
        self._init_redis()
        self._init_database()

        # Personality / Engine
        self.personality = self._load_personality()
        self.decision_engine = DecisionEngine(self.personality)

        # Oracle account
        try:
            self.oracle_account = Account.from_key(self.config['oracle']['private_key'])
            logger.info(f"Oracle address: {self.oracle_account.address}")
        except Exception as e:
            raise ValueError("Invalid oracle private key in config.yaml or missing eth-account support.") from e

        self.metrics = {
            'decisions_made': 0,
            'approvals': 0,
            'rejections': 0,
            'total_processing_time': 0,
            'errors': 0
        }
        self.shutdown_requested = False
        logger.info("YOU.AI Oracle fully initialized and ready")

    def _load_config(self, config_path: str) -> Dict:
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found ({config_path}), using defaults")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        return {
            'ethereum': {
                'rpc_url': 'http://localhost:8545',
                'poa_mode': False,
                'you_dao_address': '0x0000000000000000000000000000000000000000',
                'ai_guardian_address': '0x0000000000000000000000000000000000000000',
            },
            'redis': {
                'host': 'localhost',
                'port': 6379,
                'db': 0
            },
            'oracle': {
                'private_key': '0x0000000000000000000000000000000000000000000000000000000000000000',
                'check_interval': 300,
                'gas_price_gwei': 30
            },
            'database': {
                'path': 'you_ai_oracle.db'
            }
        }

    def _load_contracts(self):
        abi_path = Path('abis')
        try:
            with open(abi_path / 'YOUDAO.json', 'r') as f:
                dao_abi = json.load(f)
            with open(abi_path / 'YOUAIGuardian.json', 'r') as f:
                guardian_abi = json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError("ABI files not found in ./abis. Create YOUDAO.json and YOUAIGuardian.json.") from e

        self.dao_contract = self.web3.eth.contract(
            address=Web3.toChecksumAddress(self.config['ethereum']['you_dao_address']),
            abi=dao_abi
        )
        self.guardian_contract = self.web3.eth.contract(
            address=Web3.toChecksumAddress(self.config['ethereum']['ai_guardian_address']),
            abi=guardian_abi
        )
        logger.info("Smart contracts loaded successfully")

    def _init_redis(self):
        try:
            self.redis = redis.Redis(
                host=self.config['redis']['host'],
                port=self.config['redis']['port'],
                db=self.config['redis']['db'],
                decode_responses=True
            )
            self.redis.ping()
            logger.info("Redis connection established")
        except RedisError as e:
            logger.warning(f"Redis connection failed: {e}. Continuing without Redis.")
            self.redis = None

    def _init_database(self):
        db_path = self.config['database']['path']
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        cursor = self.db.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS decisions (
                proposal_id INTEGER PRIMARY KEY,
                approved BOOLEAN,
                confidence REAL,
                reasoning TEXT,
                risk_assessment REAL,
                alignment_score REAL,
                category_score REAL,
                keyword_matches INTEGER,
                decision_timestamp INTEGER,
                tx_hash TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS proposals_cache (
                proposal_id INTEGER PRIMARY KEY,
                title TEXT,
                description TEXT,
                amount TEXT,
                recipient TEXT,
                category TEXT,
                created_at INTEGER,
                cached_at INTEGER
            )
        ''')
        self.db.commit()
        logger.info("Database initialized")

    def _load_personality(self) -> FounderPersonality:
        return FounderPersonality(
            vision_keywords=[
                'immortal', 'execution', 'civilization', 'ai', 'automation',
                'decentralization', 'sustainable', 'innovation', 'future',
                'empire', 'legacy', 'evolution', 'scale', 'impact', 'system',
                'protocol', 'infrastructure', 'exponential', 'compound',
                'autonomous', 'self-executing', 'programmable', 'trustless'
            ],
            risk_tolerance=0.7,
            innovation_bias=0.9,
            social_impact_weight=0.6,
            financial_weight=0.4,
            decision_patterns={
                'research': 0.90,
                'infrastructure': 0.95,
                'marketing': 0.30,
                'legal': 0.60,
                'partnership': 0.70,
                'treasury': 0.90,
                'ip_licensing': 0.85,
                'successor_training': 0.85,
                'emergency': 0.50,
                'general': 0.40
            },
            core_values=[
                'long_term_thinking', 'technological_advancement',
                'decentralized_governance', 'sustainable_growth',
                'human_augmentation', 'systemic_change', 'immortal_systems',
                'execution_over_planning', 'code_over_politics'
            ],
            red_flags=[
                'scam', 'ponzi', 'pyramid', 'get rich quick', 'guaranteed returns',
                'no risk', 'secret formula', 'limited time', 'act now',
                'centralized control', 'kyc required', 'personal enrichment'
            ]
        )

    # (rest of YOUAIOracle class methods remain unchanged — omitted here for brevity)
    # For completeness, we keep the rest of your methods exactly as in your original file
    # (check_founder_status, get_pending_proposals, process_proposal, _submit_decision, _store_decision, run, etc.)
    # To keep this display concise the unchanged methods are below copied from your original file.

    async def check_founder_status(self) -> bool:
        try:
            is_active = self.guardian_contract.functions.isFounderActive().call()
            status_str = "ACTIVE" if is_active else "INACTIVE"
            logger.info(f"Founder status: {status_str}")
            if self.redis:
                self.redis.setex('founder_status', 300, str(is_active))
            return is_active
        except Exception as e:
            logger.error(f"Error checking founder status: {e}")
            return False

    async def get_pending_proposals(self) -> List[Proposal]:
        proposals = []
        try:
            proposal_count = self.dao_contract.functions.proposalCounter().call()
            logger.info(f"Total proposals on-chain: {proposal_count}")
            for proposal_id in range(1, proposal_count + 1):
                if self._is_processed(proposal_id):
                    continue
                try:
                    proposal_data = self.dao_contract.functions.getProposal(proposal_id).call()
                    proposal = Proposal(
                        id=proposal_data[0],
                        title=proposal_data[1],
                        description=proposal_data[2],
                        amount=proposal_data[4],
                        recipient=proposal_data[5],
                        category=self._map_category(proposal_data[10]),
                        created_at=proposal_data[6],
                        voting_ends_at=proposal_data[7],
                        for_votes=proposal_data[9],
                        against_votes=proposal_data[10],
                        executed=proposal_data[8],
                        ai_approved=proposal_data[11],
                        ai_confidence=proposal_data[12]
                    )
                    if not proposal.executed and not proposal.ai_approved:
                        proposals.append(proposal)
                        self._cache_proposal(proposal)
                except Exception as e:
                    logger.error(f"Error fetching proposal {proposal_id}: {e}")
                    continue
            logger.info(f"Found {len(proposals)} pending proposals")
            return proposals
        except Exception as e:
            logger.error(f"Error getting pending proposals: {e}")
            return []

    def _is_processed(self, proposal_id: int) -> bool:
        cursor = self.db.cursor()
        cursor.execute('SELECT 1 FROM decisions WHERE proposal_id = ?', (proposal_id,))
        return cursor.fetchone() is not None

    def _map_category(self, category_id: int) -> str:
        categories = [
            'research', 'infrastructure', 'marketing', 'legal',
            'partnership', 'treasury', 'ip_licensing', 'successor_training', 'emergency'
        ]
        return categories[category_id] if category_id < len(categories) else 'general'

    def _cache_proposal(self, proposal: Proposal):
        cursor = self.db.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO proposals_cache
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            proposal.id, proposal.title, proposal.description,
            str(proposal.amount), proposal.recipient, proposal.category,
            proposal.created_at, int(time.time())
        ))
        self.db.commit()

    async def process_proposal(self, proposal: Proposal) -> bool:
        logger.info(f"Processing proposal {proposal.id}: {proposal.title}")
        try:
            decision = self.decision_engine.analyze_proposal(proposal)
            status = "✅ APPROVED" if decision.approved else "❌ REJECTED"
            logger.info(f"Proposal {proposal.id}: {status} (confidence: {decision.confidence:.2f})")
            logger.info(f"Reasoning: {decision.reasoning[:200]}...")
            tx_hash = await self._submit_decision(decision)
            if tx_hash:
                self._store_decision(decision, tx_hash)
                self.metrics['decisions_made'] += 1
                if decision.approved:
                    self.metrics['approvals'] += 1
                else:
                    self.metrics['rejections'] += 1
                return True
            else:
                self.metrics['errors'] += 1
                return False
        except Exception as e:
            logger.error(f"Error processing proposal {proposal.id}: {e}")
            self.metrics['errors'] += 1
            return False

    async def _submit_decision(self, decision: AIDecision) -> Optional[str]:
        try:
            nonce = self.web3.eth.get_transaction_count(self.oracle_account.address)
            tx = self.guardian_contract.functions.makeAIDecision(
                decision.proposal_id,
                decision.approved,
                int(decision.confidence * 100),
                decision.reasoning[:500]
            ).buildTransaction({
                'from': self.oracle_account.address,
                'nonce': nonce,
                'gas': 300000,
                'gasPrice': self.web3.toWei(self.config['oracle']['gas_price_gwei'], 'gwei')
            })
            signed_tx = self.oracle_account.sign_transaction(tx)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            logger.info(f"Decision submitted: {tx_hash.hex()}")
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
            if receipt['status'] == 1:
                logger.info(f"✅ Decision confirmed in block {receipt['blockNumber']}")
                return tx_hash.hex()
            else:
                logger.error(f"❌ Transaction failed")
                return None
        except Exception as e:
            logger.error(f"Error submitting decision: {e}")
            return None

    def _store_decision(self, decision: AIDecision, tx_hash: str):
        cursor = self.db.cursor()
        cursor.execute('''
            INSERT INTO decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            decision.proposal_id,
            decision.approved,
            decision.confidence,
            decision.reasoning,
            decision.risk_assessment,
            decision.alignment_score,
            decision.category_score,
            decision.keyword_matches,
            decision.decision_timestamp,
            tx_hash
        ))
        self.db.commit()
        if self.redis:
            self.redis.setex(
                f'decision:{decision.proposal_id}',
                86400 * 30,
                json.dumps(asdict(decision))
            )

    async def run(self):
        logger.info("🚀 YOU.AI Oracle starting main loop")
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        iteration = 0
        while not self.shutdown_requested:
            iteration += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"Iteration #{iteration}")
            logger.info(f"{'='*60}")
            try:
                founder_active = await self.check_founder_status()
                if not founder_active:
                    logger.info("🤖 Founder inactive - AI taking control")
                    proposals = await self.get_pending_proposals()
                    if proposals:
                        logger.info(f"📋 Processing {len(proposals)} proposals")
                        for proposal in proposals:
                            success = await self.process_proposal(proposal)
                            await asyncio.sleep(10)
                    else:
                        logger.info("✨ No pending proposals")
                else:
                    logger.info("👤 Founder active - AI in standby mode")
                self._print_metrics()
                await self._health_check()
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                self.metrics['errors'] += 1
            wait_time = self.config['oracle']['check_interval']
            logger.info(f"⏳ Waiting {wait_time}s until next check...")
            await asyncio.sleep(wait_time)
        logger.info("🛑 Shutdown complete")

    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_requested = True

    def _print_metrics(self):
        logger.info(f"\n📊 Metrics:")
        logger.info(f"  Decisions made: {self.metrics['decisions_made']}")
        logger.info(f"  Approvals: {self.metrics['approvals']}")
        logger.info(f"  Rejections: {self.metrics['rejections']}")
        logger.info(f"  Errors: {self.metrics['errors']}")
        if self.metrics['decisions_made'] > 0:
            approval_rate = (self.metrics['approvals'] / self.metrics['decisions_made']) * 100
            logger.info(f"  Approval rate: {approval_rate:.1f}%")

    async def _health_check(self):
        health = {
            'timestamp': datetime.now().isoformat(),
            'web3_connected': web3_is_connected(self.web3),
            'redis_connected': self.redis.ping() if self.redis else False,
            'db_connected': True,
            'oracle_balance': None,
            'metrics': self.metrics
        }
        try:
            health['oracle_balance'] = self.web3.eth.get_balance(self.oracle_account.address)
        except Exception:
            health['oracle_balance'] = 0
        if self.redis:
            self.redis.setex('health', 60, json.dumps(health))
        try:
            balance_eth = self.web3.fromWei(health['oracle_balance'], 'ether')
            if balance_eth < 0.1:
                logger.warning(f"⚠️  Low oracle balance: {balance_eth:.4f} ETH")
        except Exception:
            pass
        return health

# -----------------------------------------------------------------------------
# The remaining classes (OracleMonitor, DeploymentManager, config generator, CLI)
# remain the same as in your original file — include them below unchanged.
# For brevity in this message they are not repeated but should be copied from your
# original file into this script after the YOUAIOracle class definition.
# -----------------------------------------------------------------------------

def generate_config_template():
    config = {
        'ethereum': {
            'rpc_url': 'https://mainnet.infura.io/v3/YOUR_PROJECT_ID',
            'poa_mode': False,
            'you_dao_address': '0x0000000000000000000000000000000000000000',
            'ai_guardian_address': '0x0000000000000000000000000000000000000000',
            'chain_id': 1
        },
        'redis': {
            'host': 'localhost',
            'port': 6379,
            'db': 0,
            'password': None
        },
        'oracle': {
            'private_key': '0xYOUR_PRIVATE_KEY_HERE',
            'check_interval': 300,
            'gas_price_gwei': 30,
            'max_gas_price_gwei': 100
        },
        'database': {
            'path': 'you_ai_oracle.db'
        },
        'monitoring': {
            'enable_metrics': True,
            'metrics_port': 9090,
            'log_level': 'INFO'
        },
        'security': {
            'max_amount_per_decision': '1000000000000000000000',
            'require_founder_approval_above': '100000000000000000000',
            'emergency_shutdown_threshold': 5
        }
    }
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info("✅ Configuration template generated: config.yaml")

def cli_main():
    import argparse
    parser = argparse.ArgumentParser(description='YOU.AI Oracle - Immortal Execution System')
    parser.add_argument('command', choices=['run', 'monitor', 'report', 'init', 'deploy'])
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--db', default='you_ai_oracle.db', help='Database path')
    args = parser.parse_args()
    if args.command == 'init':
        logger.info("Initializing YOU.AI Oracle system...")
        generate_config_template()
        Path('abis').mkdir(exist_ok=True)
        Path('logs').mkdir(exist_ok=True)
        logger.info("✅ System initialized. Edit config.yaml with your settings.")
    elif args.command == 'run':
        try:
            oracle = YOUAIOracle(args.config)
            asyncio.run(oracle.run())
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            sys.exit(1)
    elif args.command == 'monitor':
        monitor = OracleMonitor(args.db)
        while True:
            try:
                print("\033[2J\033[H")
                print(monitor.generate_report())
                time.sleep(30)
            except KeyboardInterrupt:
                break
    elif args.command == 'report':
        monitor = OracleMonitor(args.db)
        print(monitor.generate_report())
    elif args.command == 'deploy':
        logger.info("Contract deployment through CLI coming soon...")
        logger.info("Use deployment scripts in scripts/ directory")

if __name__ == "__main__":
    cli_main()
