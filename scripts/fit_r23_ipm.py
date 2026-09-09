"""Run the frozen R23 fit using HiGHS interior-point linear programming.

Only LP algorithm selection changes. Layer objectives, coefficients, data access,
and bounded least-squares fitting are the unchanged audit_r23_eyes implementation.
"""
import argparse
from scipy.optimize import linprog
import audit_r23_eyes as audit
from audit_r20_eyes import sha, write_json
from pathlib import Path
from shot_paths import add_shot_argument, from_args


def interior_point(*args,**kwargs):
    # We need an optimal feasible radiance witness, not a simplex vertex basis.
    # Crossover can dominate these highly underdetermined texture-grid LPs.
    return linprog(*args,**dict(kwargs,method='highs-ipm',
        options=dict(run_crossover='off',disp=True,ipm_optimality_tolerance=1e-10)))


def main():
    ap=argparse.ArgumentParser(description=__doc__); add_shot_argument(ap)
    args=ap.parse_args(); shot=from_args(args)
    audit.linprog=interior_point
    write_json(shot.logs/'r23_lp_algorithm.json',dict(method='highs-ipm',
        crossover=False,ipm_optimality_tolerance=1e-10,
        runner_sha256=sha(Path(__file__)),change='LP algorithm only; fixed objective, bounds and observations'))
    audit.fit_stage(shot)


if __name__=='__main__': main()
