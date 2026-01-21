
# Iodine

We release the code, dataset, and results of work in this repository.

## 📁 Directory Structure

```
.
├── code        # The source code of the analysis pipeline
├── data        # The dataset of analyzed smart contracts
├── results     # The results of the analysis
```

## 🛠️ Setup

### 1. Install Dependencies

- Python 3.8+
- [Mythril](https://github.com/ConsenSys/mythril) 
- [ANTLR 4](https://github.com/antlr/antlr4)
- [NuSMV](http://nusmv.fbk.eu/) (v2.6.0+)

### 2. Generate DSL Parser

Run ANTLR to generate the Python parser from the grammar:

```bash
antlr4 -Dlanguage=Python3 duduDSL.g4
```

This generates `duduDSLLexer.py`, `duduDSLParser.py`, etc., used to parse your custom DSL rules.

> ✅ The generated parser files are included — you only need to run this if modifying `duduDSL.g4`.

## 🔁 Usage Pipeline

### Step 1: Run Mythril to Get Execution Trace

Analyze a contract using its bytecode or address:

```bash
myth analyze DApp.bytecode --json > trace.json
```

This generates a symbolic execution trace with paths, constraints, and state changes.

### Step 2: Write Your DSL Rules for Semantic Abstraction

Create a `finance_rules.dsl` file to define how low-level storage operations map to high-level financial logic. You must write these rules based on the target contract’s logic.

### Step 3: Parse Trace and Build Model Using DSL

Run the main analysis script with your DSL file:

```bash
python DefiCheck3.py trace.json finance_rules.dsl
```

This:
- Parses the Mythril trace
- Applies your DSL rules to lift operations
- Builds a symbolic dependency graph with financial semantics
- Outputs an intermediate model (e.g., `defi_model.pkl`)

### Step 4: Generate NuSMV Model

Convert the abstracted graph into a model for verification:

```bash
python model.py defi_model.pkl --ltl props.ltl > FBLM.smv
```

The resulting `FBLM.smv` encodes:
- State variables (from DSL)
- Transitions (from contract behavior)
- LTL properties (in props.ltl)

### Step 5: Run NuSMV Model Checker

Verify the model:

```bash
NuSMV FBLM.smv
```

Check output for:
- `is true` → Property satisfied
- `is false` + counterexample → Bug found (e.g., invalid state reachable)

---

## 📝 Notes

- You **must write your own DSL rules** to capture the financial logic of the target contract.
- The DSL supports: `def`, `rule`, `on SLOAD/SSTORE`, `ASSERT`, `EVENT`, and symbolic expressions.
- Use `props.ltl` to specify temporal properties like `G !(balance < 0)` or `F (supply > 0)`.


