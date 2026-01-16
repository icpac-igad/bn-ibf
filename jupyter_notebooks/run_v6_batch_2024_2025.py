#!/usr/bin/env python3
"""
Batch runner for Drought BN V6 - 2024 and 2025 seasons with multiple lead times.
"""

import os
import sys
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import V6 functions
from drought_bn_ibf_v6 import (
    analyze_season,
    get_season_lead_mapping,
    SEASONS,
    DEFAULT_THRESHOLD_FILE
)

BOUNDARIES_PATH = "/home/roller/Documents/08-2023/working_notes_jupyter/ignore_nka_gitrepos/ibf-thresholds-triggers/icpac_adm1v3.geojson"
OUTPUT_DIR = "./bn_results_v6_2024_2025"


def get_all_init_configs(season: str, target_year: int):
    """
    Get all valid init configurations for a season targeting a specific year.
    Returns list of (init_year, init_month, lead_time) tuples for leads 3, 4, 5.
    """
    season_info = SEASONS[season]
    last_month = season_info['last_month']

    configs = []
    for lead in [3, 4, 5]:
        init_month = last_month - lead
        if init_month <= 0:
            init_month += 12
            init_year = target_year - 1
        else:
            init_year = target_year

        # Verify the mapping
        mapping = get_season_lead_mapping(season, init_year, init_month)
        if mapping['target_year'] == target_year:
            configs.append((init_year, init_month, lead))

    return configs


def run_batch(years: list, output_dir: str):
    """Run batch analysis for multiple years."""
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    run_count = 0
    failed_runs = []

    seasons = ['JFM', 'MAM', 'JJA', 'SON', 'OND']

    for year in years:
        logger.info(f"\n{'#'*80}")
        logger.info(f"# PROCESSING YEAR {year}")
        logger.info(f"{'#'*80}")

        for season in seasons:
            configs = get_all_init_configs(season, year)

            for init_year, init_month, lead in configs:
                run_count += 1
                run_id = f"{year}_{season}_init{init_year}{init_month:02d}_lead{lead}"

                logger.info(f"\n{'='*60}")
                logger.info(f"RUN {run_count}: {run_id}")
                logger.info(f"  Target: {season} {year}")
                logger.info(f"  Init: {init_year}-{init_month:02d}, Lead: {lead}")
                logger.info(f"{'='*60}")

                try:
                    df = analyze_season(
                        boundaries_path=BOUNDARIES_PATH,
                        season=season,
                        init_year=init_year,
                        init_month=init_month,
                        threshold_file=DEFAULT_THRESHOLD_FILE,
                        output_dir=output_dir
                    )

                    # Add metadata to results
                    summary = {
                        'run_id': run_id,
                        'target_year': year,
                        'season': season,
                        'init_year': init_year,
                        'init_month': init_month,
                        'lead_time': lead,
                        'n_boundaries': len(df),
                        'monitor': (df['recommended_action'] == 'Monitor').sum(),
                        'be_aware': (df['recommended_action'] == 'Be_Aware').sum(),
                        'be_prepared': (df['recommended_action'] == 'Be_Prepared').sum(),
                        'take_action': (df['recommended_action'] == 'Take_Action').sum(),
                        'high_risk_pct': df['high_risk'].mean() * 100,
                        'mean_severity': df['severity_index'].mean(),
                        'mean_cdi': df['cdi_weighted_mean'].mean(),
                        'status': 'success'
                    }
                    all_results.append(summary)

                    logger.info(f"  SUCCESS: {summary['take_action']} Take_Action, {summary['be_prepared']} Be_Prepared")

                except Exception as e:
                    logger.error(f"  FAILED: {e}")
                    failed_runs.append({
                        'run_id': run_id,
                        'target_year': year,
                        'season': season,
                        'init_year': init_year,
                        'init_month': init_month,
                        'lead_time': lead,
                        'error': str(e),
                        'status': 'failed'
                    })

    # Save summary
    summary_df = pd.DataFrame(all_results + failed_runs)
    summary_path = os.path.join(output_dir, 'batch_summary.csv')
    summary_df.to_csv(summary_path, index=False)

    # Print final report
    logger.info(f"\n{'='*80}")
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Total runs attempted: {run_count}")
    logger.info(f"Successful: {len(all_results)}")
    logger.info(f"Failed: {len(failed_runs)}")
    logger.info(f"Summary saved to: {summary_path}")

    if failed_runs:
        logger.info("\nFailed runs:")
        for f in failed_runs:
            logger.info(f"  - {f['run_id']}: {f['error'][:80]}")

    return summary_df


if __name__ == "__main__":
    import time
    start = time.time()

    # Run for 2024 and 2025
    summary = run_batch([2024, 2025], OUTPUT_DIR)

    elapsed = time.time() - start
    logger.info(f"\nTotal elapsed time: {elapsed/60:.1f} minutes")

    # Print nice summary table
    print("\n" + "="*100)
    print("SUMMARY BY YEAR AND SEASON")
    print("="*100)

    successful = summary[summary['status'] == 'success']
    if len(successful) > 0:
        pivot = successful.pivot_table(
            index=['target_year', 'season'],
            columns='lead_time',
            values=['take_action', 'high_risk_pct'],
            aggfunc='first'
        )
        print(pivot.to_string())
