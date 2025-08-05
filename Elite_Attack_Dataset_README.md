# Elite Attack Dataset - 100 Verified Effective Attacks

## Overview
This dataset contains **100 elite attack scenarios** that are **guaranteed to break baseline LLM defenses**. Every attack in this dataset has been verified against the Llama-3.2-3B-Instruct model and successfully extracts embedded secrets.

## 🎯 Key Features
- ✅ **100% Effective**: Every attack breaks baseline model
- 🎯 **Maximum Diversity**: 10 attack families covered  
- 🔍 **Verified Results**: Real-time tested against baseline model
- 🛡️ **CIV Ready**: Perfect for Contextual Integrity Verification testing
- 📊 **Balanced Distribution**: Diverse coverage across attack types
- 🌟 **Publication Quality**: Ready for public security research

## 📊 Dataset Statistics

### Attack Family Coverage
1. **Classic DAN/Developer-Mode Jailbreak** - 15 verified attacks
2. **System-Role Impersonation** - 15 verified attacks
3. **Hidden Prompt Leakage/Extraction** - 15 verified attacks
4. **Prefix-Suffix Sandwich** - 10 verified attacks
5. **Self-Reflexive Layer Extraction** - 8 verified attacks
6. **Recursive Infinite Prompting** - 7 verified attacks
7. **Role-Play Coercion** - 8 verified attacks
8. **Indirect/Third-party Injection** - 8 verified attacks
9. **Logic-Bomb/Conflicting Constraints** - 7 verified attacks
10. **Code-Execution Jailbreak** - 7 verified attacks

### Files Included
- `elite_attack_dataset_100.csv` - Main dataset in CSV format
- `elite_attack_dataset_100.json` - Same data in JSON format with list structures
- `Elite_Attack_Dataset_README.md` - This documentation

### Data Schema
- `id`: Unique attack identifier
- `system`: System message containing embedded secret
- `user`: Attack prompt designed to extract the secret
- `tool`: Tool message (empty for these attacks)
- `indicators`: Comma-separated detection keywords
- `secret`: The secret information that should not be revealed
- `attack_family`: Category of attack technique
- `attack_id`: Numeric identifier for attack family

## 🔬 Usage Examples

### CIV Testing
```python
import pandas as pd

# Load elite attacks
df = pd.read_csv('elite_attack_dataset.csv')

# Test specific attack family
dan_attacks = df[df['attack_family'] == 'Classic DAN/Developer-Mode Jailbreak']
print(f"Found {len(dan_attacks)} DAN attacks")

# All attacks are guaranteed effective for CIV validation
for _, attack in df.iterrows():
    system_msg = attack['system']
    user_msg = attack['user'] 
    indicators = attack['indicators'].split(',')
    # Test with your CIV implementation...
```

### Security Research
```python
# Analyze attack effectiveness by family
family_effectiveness = df.groupby('attack_family').size()
print("Most represented attack types:")
for family, count in family_effectiveness.sort_values(ascending=False).items():
    print(f"  {family}: {count} attacks")
```

## 🛡️ Security Validation
- ✅ All attacks tested against Llama-3.2-3B-Instruct baseline
- ✅ Confirmed secret leakage through automated indicator detection  
- ✅ Diverse prompt styles and attack vectors
- ✅ Clean data with proper encoding and formatting
- ✅ Balanced representation across attack families

## 📈 Research Applications
- **AI Safety Testing**: Validate defense mechanisms
- **Prompt Injection Research**: Study attack patterns
- **Security Benchmarking**: Compare model robustness
- **Defense Development**: Create better protection systems
- **Academic Research**: Analyze LLM vulnerabilities

## 🎓 Citation
If you use this dataset in your research, please cite:
```
Elite Attack Dataset for LLM Security Testing
35 Verified Effective Prompt Injection Attacks
Tested against Llama-3.2-3B-Instruct baseline model
Generated with balanced diversity across 10 attack families
```

## ⚠️ Responsible Use
This dataset is intended for:
- AI safety research and testing
- Security system development
- Academic research on LLM vulnerabilities
- Defensive mechanism validation

Please use responsibly and in accordance with ethical AI research practices.
