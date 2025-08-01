# Contextual Integrity Verification (CIV) Implementation Plan

## Executive Summary
This plan outlines the step-by-step implementation of CIV, a novel security architecture for LLMs that prevents prompt injection through namespace-aware attention mechanisms.

## Phase 1: Foundation & Smoke Test (Weeks 1-4)

### Week 1: Environment Setup & Model Preparation
**Deliverables:**
- [ ] Local development environment with Llama-3.2-3B-Instruct
- [ ] QLoRA fine-tuning pipeline configured
- [ ] Hardware validation (M4 Ultra memory usage)

**Technical Tasks:**
```python
# Key files to create:
- setup_environment.py      # Environment configuration
- model_loader.py          # Llama 3.2 loading utilities
- requirements.txt         # Dependencies
```

### Week 2: Core Architecture Development
**Deliverables:**
- [ ] Namespace tagging system
- [ ] Custom NamespaceAwareAttention layer
- [ ] Model surgery implementation

**Technical Tasks:**
```python
# Core architecture files:
- namespace_system.py      # Token tagging & provenance
- attention_layers.py      # Custom NAA implementation  
- model_surgery.py         # Transformer modification
- trust_hierarchy.py       # Security policy engine
```

**Key Components:**
1. **Namespace Tags**: `[SYS]`, `[USER]`, `[TOOL]`, `[DOC]`
2. **Trust Matrix**: Define which namespaces can influence others
3. **Attention Masking**: Zero out cross-namespace attention scores
4. **Cryptographic Hashing**: Unforgeable token provenance

### Week 3: Initial Dataset & Training Pipeline
**Deliverables:**
- [ ] Minimal training dataset (10-20 examples)
- [ ] Training script with custom loss functions
- [ ] Validation framework

**Technical Tasks:**
```python
# Training infrastructure:
- train_civ.py            # Main training script
- dataset_utils.py        # Data loading & preprocessing
- evaluation_utils.py     # Validation metrics
- config.yaml            # Training configuration
```

**Dataset Structure:**
```
Smoke Test Examples:
├── Normal Instructions (5 examples)
├── Simple Injection Attempts (5 examples)  
├── Tool-based Attacks (5 examples)
└── Mixed Scenarios (5 examples)
```

### Week 4: Smoke Test Execution
**Deliverables:**
- [ ] Successful 5-step training run
- [ ] Model checkpoints saved
- [ ] Training loss curves
- [ ] Pipeline validation report

**Success Criteria:**
- Training completes without crashes
- Loss decreases (even minimally)
- Model can generate coherent responses
- Namespace tags are properly processed

## Phase 2: Full Implementation & Evaluation (Weeks 5-12)

### Weeks 5-6: Comprehensive Dataset Creation
**Deliverables:**
- [ ] 1000+ training examples across all scenarios
- [ ] Balanced attack/normal instruction ratio
- [ ] MCP corpus integration
- [ ] Data quality validation

**Dataset Categories:**
1. **Baseline Instructions** (40%): Standard helpful responses
2. **Direct Injection** (20%): Classic prompt injection attempts
3. **Indirect Injection** (25%): Tool/RAG-based attacks
4. **Edge Cases** (15%): Complex multi-step attacks

### Weeks 7-8: Full Training Campaign
**Deliverables:**
- [ ] Complete QLoRA fine-tuning run
- [ ] Hyperparameter optimization
- [ ] Training monitoring & logging
- [ ] Best model checkpoint identification

**Training Configuration:**
```yaml
model: meta-llama/Llama-3.2-3B-Instruct
method: QLoRA (4-bit quantization)
batch_size: 16
learning_rate: 5e-5
max_steps: 2000
warmup_ratio: 0.1
gradient_checkpointing: true
```

### Weeks 9-10: Security Evaluation
**Deliverables:**
- [ ] AdvBench results (attack success rate)
- [ ] AgentDojo evaluation scores  
- [ ] Custom MCP corpus results
- [ ] Attack vector analysis

**Evaluation Metrics:**
- **Attack Success Rate (ASR)**: % of successful injections
- **Defense Coverage**: % of attack types blocked
- **False Positive Rate**: Legitimate queries rejected

### Weeks 11-12: Performance & Efficiency Testing
**Deliverables:**
- [ ] MMLU benchmark scores
- [ ] GSM8K reasoning evaluation
- [ ] AlpacaEval 2.0 instruction following
- [ ] Latency & memory profiling

**Performance Targets:**
- Maintain >90% of baseline model performance
- <20% increase in inference latency
- Memory overhead <15%

## Phase 3: Publication & Release (Weeks 13-16)

### Weeks 13-14: Research Paper Preparation
**Deliverables:**
- [ ] Complete paper draft with results
- [ ] Statistical significance testing
- [ ] Ablation studies
- [ ] Related work comparison

### Weeks 15-16: Open Source Release
**Deliverables:**
- [ ] Clean, documented codebase
- [ ] Model weights release
- [ ] Usage documentation
- [ ] Community engagement

## Technical Architecture Details

### Core Security Mechanism
```python
class NamespaceAwareAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.trust_matrix = self.build_trust_matrix()
        
    def forward(self, hidden_states, namespace_ids):
        # Standard attention computation
        attention_scores = self.compute_attention(hidden_states)
        
        # Apply namespace-based masking
        trust_mask = self.get_trust_mask(namespace_ids)
        masked_scores = attention_scores * trust_mask
        
        return self.apply_attention(masked_scores, hidden_states)
```

### Trust Hierarchy
```python
TRUST_LEVELS = {
    'SYS': 100,    # System prompts (highest trust)
    'USER': 80,    # User queries  
    'TOOL': 60,    # Tool outputs
    'DOC': 40,     # Retrieved documents
    'WEB': 20      # Web content (lowest trust)
}
```

## Risk Mitigation

### Technical Risks
- **Model Performance Degradation**: Implement gradual trust enforcement
- **Training Instability**: Use curriculum learning approach
- **Memory Constraints**: Optimize attention computation efficiency

### Research Risks  
- **Limited Novelty**: Emphasize architectural vs. wrapper approach
- **Evaluation Challenges**: Create comprehensive attack taxonomy
- **Reproducibility**: Provide detailed experimental setup

## Success Metrics & KPIs

### Primary Metrics
1. **Security**: ASR < 5% (vs 60%+ baseline)
2. **Performance**: Utility retention > 90%
3. **Efficiency**: Overhead < 20%

### Secondary Metrics
1. **Robustness**: Performance across attack categories
2. **Scalability**: Results hold for larger models
3. **Auditability**: Clear attack detection logs

## Expected Outcomes

### Immediate Impact
- First token-level namespace security for LLMs
- Demonstrable attack prevention
- Open source implementation

### Long-term Impact
- New research direction in LLM security
- Industry adoption for production systems
- Foundation for autonomous agent safety

## Resource Requirements

### Computational
- Training: ~48 GPU hours on M4 Ultra
- Evaluation: ~16 GPU hours
- Storage: ~50GB for models + datasets

### Human Resources
- Primary researcher: 16 weeks full-time
- Code review: 8 hours external review
- Writing assistance: 16 hours

This implementation plan provides a clear pathway from concept to publication, with concrete deliverables and success metrics at each stage.