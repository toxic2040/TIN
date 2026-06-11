# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in TIN, please report it
responsibly by emailing **jcouncilman2040@gmail.com**.

Do not open a public issue for security vulnerabilities.

## Scope

TIN is a research simulation tool (TRL 2-3). It is not designed for
production deployment. Security concerns most likely to be relevant:

- Arbitrary code execution via crafted input files (contact plans, traces)
- Dependency vulnerabilities in NumPy or optional packages
- Exposure of SPICE kernel paths or local filesystem information

## Response

I will acknowledge receipt within 72 hours and provide an update within
two weeks. Fixes will be released as a new commit on `main`.
