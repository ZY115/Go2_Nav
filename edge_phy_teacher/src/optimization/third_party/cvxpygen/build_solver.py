#!/usr/bin/env python3
# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

import os
import sys
import time
import shutil
import importlib
import logging
import numpy as np
import cvxpy as cp
from cvxpygen import cpg

from ddp_lmi_solver import cvxpy_ddp

np.set_printoptions(suppress=True)

solver = 'ECVXCONE'
CODE_DIR = 'ECVXCONE'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def deploy_solver(src_dir):
    """
    Deploy the generated C code to the target directory.
    Print confirmation for each copied file.
    """

    # Define source files
    files_to_copy = [
        (os.path.join(src_dir, 'c', 'include', 'cpg_solve.h'),
         os.path.join(BASE_DIR, '../../../../include/ecvxcone')),

        (os.path.join(src_dir, 'c', 'include', 'cpg_workspace.h'),
         os.path.join(BASE_DIR, '../../../../include/ecvxcone')),

        (os.path.join(src_dir, 'c', 'src', 'cpg_solve.c'),
         os.path.join(BASE_DIR, '../../ecvxcone')),

        (os.path.join(src_dir, 'c', 'src', 'cpg_workspace.c'),
         os.path.join(BASE_DIR, '../../ecvxcone')),
    ]

    for src_file, dst_folder in files_to_copy:
        if not os.path.exists(src_file):
            raise FileNotFoundError(f"Source file not found: {src_file}")

        os.makedirs(dst_folder, exist_ok=True)
        shutil.copy2(src_file, dst_folder)
        print(f"✔ Copied: {src_file} → {dst_folder}")

    print(f"\n🎉 Successfully deployed {solver} solver.")


def generate_solver(code_dir, solver):
    '''
    Generate C Code for the DDP solver
    '''
    problem, _, _, _ = cvxpy_ddp()
    
    # generate code
    cpg.generate_code(problem, code_dir=code_dir, solver=solver, wrapper=False)

if __name__ == "__main__":

    solver_dir = os.path.join(BASE_DIR, CODE_DIR)
    generate_solver(code_dir=solver_dir, solver=solver)
    deploy_solver(src_dir=solver_dir)
