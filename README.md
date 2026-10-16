# ESS Advisor

Implementation repository for the Extended Statistics Selection (ESS) research project. The project will keep three conceptually separate concerns:

- admissibility rules for valid estimation schemes;
- estimation models and calibration data; and
- maintenance-cost models and subsequent optimization algorithms.

Related repositories:

- [ess-postgresql](https://github.com/1951123/ess-postgresql): PostgreSQL source and future ESS integration work.
- [ess-research](https://github.com/1951123/ess-research): research formulations, analyses, and reproducible studies.

## Synthetic prototype

The standalone prototype is installed and tested in the repository-local
virtual environment:

    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
    .venv/bin/python -m pip install -e .
    .venv/bin/pytest -q

It provides an exact Fraction-based exhaustive reference for small instances,
a static strengthened conditional MILP, and an integer Master--Oracle
prototype using SciPy optimize.milp with HiGHS. MIP statuses are reported
explicitly; a time-limited incumbent is never labeled optimal.
