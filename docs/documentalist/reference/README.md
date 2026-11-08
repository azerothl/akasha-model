---
title: Akasha Model
description: Akasha Model is a PyTorch package for choosing among typed options in a single forward pass.
---

# Akasha Model

Akasha Model is a PyTorch package for choosing among typed options in a single forward pass. A question can request a choice, a score, or the abstention label. The tool gate uses these outputs to decide how to handle a proposed call.

## Input and contracts

The README defines the JSONL schema and training entry points. The contracts describe the shared mix descriptors and presets. For tool calling, the model produces a plan or gate result and the caller owns tool execution.

## Sources

- [Model and JSONL format](https://github.com/azerothl/akasha-model/blob/main/README.md)
- [Contracts](https://github.com/azerothl/akasha-model/blob/main/docs/contracts/README.md)
- [Tool gate](https://github.com/azerothl/akasha-model/blob/main/docs/using-the-tool-gate.md)
