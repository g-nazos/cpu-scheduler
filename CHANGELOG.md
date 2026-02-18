# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Common Changelog](https://common-changelog.org/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed

- Integer program solver and all optimal-solution comparison (IP module, optimal metrics, gap/ratio in metrics, plots, and CLI)
- scipy dependency (was only used by IP solver)

### Added

- Initial project structure
- Core models: Agent, Slot, Market
- Ascending auction algorithm (Figure 2.7 from Section 2.3.3)
- Equilibrium verification (Definition 2.3.11)
- Book examples reproduction (8-slot, 2-slot problems)
- Visualization suite for price evolution and allocation
- Experimental analysis of convergence behavior
