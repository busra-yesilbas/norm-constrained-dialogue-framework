import sys, json, pathlib
sys.path.insert(0, 'src')

ep_files = sorted(pathlib.Path('results/sample_runs').glob('episode_*.json'))
print(f"=== SAVED TRANSCRIPT FILES ({len(ep_files)} episodes) ===")
for f in ep_files:
    with open(f) as fh:
        ep = json.load(fh)
    scenario = ep.get('scenario', {})
    turns = [t for t in ep['turns'] if t['speaker'] == 'agent']
    metrics = [t['turn_metrics'] for t in turns if t.get('turn_metrics')]
    mean_comp = round(sum(m['composite_score'] for m in metrics) / len(metrics), 3) if metrics else 'N/A'
    print(f"  {f.name}")
    print(f"    strategy        = {ep['agent_strategy']}")
    print(f"    profile         = {scenario.get('respondent_profile')}")
    print(f"    scenario_type   = {scenario.get('scenario_type')}")
    print(f"    sensitivity     = {scenario.get('sensitivity_level')}")
    print(f"    agent_turns     = {ep['total_turns']}")
    print(f"    mean_composite  = {mean_comp}")
    print(f"    final_trust     = {round(ep.get('final_trust_level', 0), 3)}")
    print(f"    final_stress    = {round(ep.get('final_stress_level', 0), 3)}")
    print()
