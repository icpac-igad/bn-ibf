#!/usr/bin/env python3
"""
Run BN analysis for all seasons and lead times for 2024.

5 seasons × 4 lead times = 20 runs
"""

import sys
import os

# Add the script directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from drought_bn_ibf_v5 import analyze_season, logger
import pandas as pd
from pathlib import Path

# Configuration
BOUNDARIES_PATH = "/srv/empirical_probability_output/icpac_adm1v3.geojson"
SERVICE_ACCOUNT = "/home/roller/Documents/08-2023/impact_weather_icpac/lab/icpac_gcp/e4drr/gcp-coiled-sa-20250310/coiled-data-e4drr_202505.json"
OUTPUT_DIR = Path(script_dir) / "bn_results_2024"

# All season configurations for 2024
# Format: (season, target_year, [(init_year, init_month, lead), ...])
SEASON_CONFIGS = {
    'JFM': {
        'target_year': 2024,
        'target_month': 3,  # March
        'inits': [
            (2023, 10, 5),  # Oct 2023 → Mar 2024 = 5 months
            (2023, 11, 4),  # Nov 2023 → Mar 2024 = 4 months
            (2023, 12, 3),  # Dec 2023 → Mar 2024 = 3 months
            (2024, 1, 2),   # Jan 2024 → Mar 2024 = 2 months
        ]
    },
    'MAM': {
        'target_year': 2024,
        'target_month': 5,  # May
        'inits': [
            (2023, 12, 5),  # Dec 2023 → May 2024 = 5 months
            (2024, 1, 4),   # Jan 2024 → May 2024 = 4 months
            (2024, 2, 3),   # Feb 2024 → May 2024 = 3 months
            (2024, 3, 2),   # Mar 2024 → May 2024 = 2 months
        ]
    },
    'JJA': {
        'target_year': 2024,
        'target_month': 8,  # August
        'inits': [
            (2024, 3, 5),   # Mar 2024 → Aug 2024 = 5 months
            (2024, 4, 4),   # Apr 2024 → Aug 2024 = 4 months
            (2024, 5, 3),   # May 2024 → Aug 2024 = 3 months
            (2024, 6, 2),   # Jun 2024 → Aug 2024 = 2 months
        ]
    },
    'SON': {
        'target_year': 2024,
        'target_month': 11,  # November
        'inits': [
            (2024, 6, 5),   # Jun 2024 → Nov 2024 = 5 months
            (2024, 7, 4),   # Jul 2024 → Nov 2024 = 4 months
            (2024, 8, 3),   # Aug 2024 → Nov 2024 = 3 months
            (2024, 9, 2),   # Sep 2024 → Nov 2024 = 2 months
        ]
    },
    'OND': {
        'target_year': 2024,
        'target_month': 12,  # December
        'inits': [
            (2024, 7, 5),   # Jul 2024 → Dec 2024 = 5 months
            (2024, 8, 4),   # Aug 2024 → Dec 2024 = 4 months
            (2024, 9, 3),   # Sep 2024 → Dec 2024 = 3 months
            (2024, 10, 2),  # Oct 2024 → Dec 2024 = 2 months
        ]
    },
}


def main():
    """Run all season/lead combinations."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results_summary = []
    total_runs = sum(len(cfg['inits']) for cfg in SEASON_CONFIGS.values())
    current_run = 0

    print("="*80)
    print(f"DROUGHT BN ANALYSIS - ALL 2024 SEASONS")
    print(f"Total runs: {total_runs} (5 seasons × 4 lead times)")
    print("="*80)

    for season in ['JFM', 'MAM', 'JJA', 'SON', 'OND']:
        config = SEASON_CONFIGS[season]
        print(f"\n{'#'*80}")
        print(f"# SEASON: {season} 2024 (Target: Month {config['target_month']})")
        print(f"{'#'*80}")

        for init_year, init_month, expected_lead in config['inits']:
            current_run += 1
            print(f"\n[{current_run}/{total_runs}] {season} - Init: {init_year}-{init_month:02d} (Lead {expected_lead})")

            try:
                df = analyze_season(
                    boundaries_path=BOUNDARIES_PATH,
                    season=season,
                    init_year=init_year,
                    init_month=init_month,
                    service_account_file=SERVICE_ACCOUNT,
                    n_cdi_months=6,
                    output_dir=str(OUTPUT_DIR),
                    include_confidence_node=True
                )

                # Get output filename
                output_file = OUTPUT_DIR / f"drought_bn_v5_{season}_{init_year}_{init_month:02d}.csv"

                # Compute summary stats
                n_boundaries = len(df)
                n_high_risk = df['high_risk'].sum()
                action_dist = df['recommended_action'].value_counts().to_dict()

                results_summary.append({
                    'season': season,
                    'init_year': init_year,
                    'init_month': init_month,
                    'lead_time': expected_lead,
                    'output_file': output_file.name,
                    'n_boundaries': n_boundaries,
                    'n_high_risk': n_high_risk,
                    'n_monitor': action_dist.get('Monitor', 0),
                    'n_be_aware': action_dist.get('Be_Aware', 0),
                    'n_be_prepared': action_dist.get('Be_Prepared', 0),
                    'n_take_action': action_dist.get('Take_Action', 0),
                    'status': 'SUCCESS'
                })

                print(f"  ✓ Created: {output_file.name}")
                print(f"    Boundaries: {n_boundaries}, High Risk: {n_high_risk}")

            except Exception as e:
                print(f"  ✗ FAILED: {e}")
                results_summary.append({
                    'season': season,
                    'init_year': init_year,
                    'init_month': init_month,
                    'lead_time': expected_lead,
                    'output_file': 'N/A',
                    'n_boundaries': 0,
                    'n_high_risk': 0,
                    'n_monitor': 0,
                    'n_be_aware': 0,
                    'n_be_prepared': 0,
                    'n_take_action': 0,
                    'status': f'FAILED: {str(e)[:50]}'
                })

    # Save summary
    summary_df = pd.DataFrame(results_summary)
    summary_file = OUTPUT_DIR / "2024_all_seasons_summary.csv"
    summary_df.to_csv(summary_file, index=False)

    # Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    success_count = sum(1 for r in results_summary if r['status'] == 'SUCCESS')
    print(f"\nCompleted: {success_count}/{total_runs} runs successful")

    if success_count > 0:
        print(f"\nOutput files created in: {OUTPUT_DIR}")
        print("\nSummary by season:")
        for season in ['JFM', 'MAM', 'JJA', 'SON', 'OND']:
            season_results = [r for r in results_summary if r['season'] == season and r['status'] == 'SUCCESS']
            if season_results:
                total_high_risk = sum(r['n_high_risk'] for r in season_results)
                print(f"  {season}: {len(season_results)}/4 runs, Total high-risk: {total_high_risk}")

    print(f"\nSummary saved: {summary_file}")

    return summary_df


if __name__ == "__main__":
    main()
