# Dataset-Aware Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make report generation dataset-aware so ISIC2018 and GlaS runs do not overwrite each other and are not mixed into the same summary by default.

**Architecture:** Thread a `dataset` selector through the report script and reporting module. Use dataset-specific artifact roots and dataset-specific default output subdirectories, and carry a `dataset` column in the run-level and method-level outputs so combined reporting remains possible later.

**Tech Stack:** Python, argparse, pandas, pytest

---
