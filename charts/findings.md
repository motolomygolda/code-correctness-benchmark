# Findings Report

## RQ1: Does pass@1 decline monotonically with difficulty?

**GPT-3.5-Turbo**: pass@1 across Easy/Medium/Hard = 0.900 / 0.744 / 0.811 — NOT monotonically declining.
**CodeLlama-7B-Instruct**: pass@1 across Easy/Medium/Hard = 0.878 / 0.800 / 0.789 — monotonically declining.

## RQ2: Does failure composition shift across tiers?

**GPT-3.5-Turbo**:
  - Easy: Syntactic 22%, Runtime 33%, Algorithmic 44%
  - Medium: Syntactic 22%, Runtime 39%, Algorithmic 39%
  - Hard: Syntactic 29%, Runtime 47%, Algorithmic 24%

**CodeLlama-7B-Instruct**:
  - Easy: Syntactic 27%, Runtime 73%
  - Medium: Syntactic 22%, Runtime 50%, Algorithmic 28%
  - Hard: Syntactic 16%, Runtime 68%, Algorithmic 16%

## RQ3: Does pass@3 preferentially recover specific failure types?

**GPT-3.5-Turbo**:
  - Syntactic: 11/12 recovered (92%)
  - Algorithmic: 13/17 recovered (76%)
  - Runtime: 19/20 recovered (95%)
**CodeLlama-7B-Instruct**:
  - Runtime: 27/30 recovered (90%)
  - Syntactic: 10/10 recovered (100%)
  - Algorithmic: 5/8 recovered (62%)

## RQ4: Model-scale effect on failure distribution?

**GPT-3.5-Turbo**: Syntactic 24%, Runtime 41%, Algorithmic 35%
**CodeLlama-7B-Instruct**: Syntactic 21%, Runtime 62%, Algorithmic 17%

Cross-model Jaccard similarity: 0.100
