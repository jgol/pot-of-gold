"""Convert H3 risk scores and rejection decisions into portfolio allocations."""
import math

import numpy as np

POLICIES = ['court', 'deploy_all', 'evidence_of_skill', 'single_llm']
LABELS = ['Court', 'Deploy-all · full size', 'Evidence of skill · matched sizes',
          'Single LLM · same formula']


def size(record, court=False):
    """Apply the published rejection veto and inverse-risk allocation formula."""
    decision = record.get('decision')
    if decision not in ('WILL_GENERALIZE', 'UNSURE', 'WONT_GENERALIZE'):
        raise ValueError(f'Invalid decision: {decision}')
    if decision == 'WONT_GENERALIZE':
        return 0.0
    score = (((record.get('disagreement') or {}).get('exploratory_ranking') or {})
             .get('median_risk_score')) if court else record.get('risk_score')
    if (isinstance(score, bool) or not isinstance(score, (int, float))
            or not math.isfinite(score) or not 0 <= score <= 100):
        raise ValueError('Missing or invalid risk score')
    return (100.0 - score) / 100.0


def allocations(keys, cohort):
    """Build court, deploy-all, matched-size and single-LLM allocations.

    Assign the court's allocation sizes in descending order to strategies
    ranked by training net P&L for the matched comparator.
    """
    court = np.array([size(cohort[k]['court'], court=True) for k in keys])
    single = np.array([size(cohort[k]['single']) for k in keys])
    ranking = sorted(range(len(keys)), key=lambda i: (-cohort[keys[i]]['train_net'], keys[i]))
    rule = np.zeros(len(keys))
    rule[ranking] = np.sort(court)[::-1]
    # Preserve the court's allocation sizes, including zero allocations.
    np.testing.assert_array_equal(np.sort(rule), np.sort(court))
    return np.stack([court, np.ones(len(keys)), rule, single])
