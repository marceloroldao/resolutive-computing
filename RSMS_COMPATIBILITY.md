# RSMS Compatibility

**Repository:** `marceloroldao/resolutive-computing`  
**RSMS compatibility:** `1.0-rc.1`  
**Normative source:** `marceloroldao/resolutive-science/docs/RSMS/RSMS_v1.0.md`

## Scope

This repository is a computational implementation project that follows the terminology, scientific-method requirements, validation principles, versioning rules, and traceability conventions of the Resolutive Science Mathematical Specification (RSMS).

The optimization algorithms implemented here are computational research objects. Their benchmark performance shall not be interpreted as empirical validation of Resolutive Physics.

## Conformance rules

1. New resolutive terminology should first be checked against the RSMS registry and definitions.
2. Domain-specific concepts shall not silently redefine RSMS symbols or identifiers.
3. Reproducible numerical claims shall record code version, benchmark configuration, random seeds, dimensions, evaluation budgets, and comparison methods.
4. Favorable, neutral, and negative results shall be retained under the same reporting standard.
5. Any intentional departure from RSMS 1.0-rc.1 shall be documented before it is treated as part of the project specification.

## Upgrade policy

When the normative RSMS version changes, compatibility shall be reviewed explicitly. This file must be updated only after the relevant terminology, interfaces, tests, and documentation have been audited against the new RSMS version.
