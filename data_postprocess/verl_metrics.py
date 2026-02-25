#!/usr/bin/env python3
"""
Wrapper to run verl.trainer.main_ppo with reward_std_patch applied.
Usage: python run_verl_with_std_patch.py [verl args...]
"""
import reward_std_patch  # Apply patch BEFORE verl imports
from verl.trainer.main_ppo import main

if __name__ == "__main__":
    main()
