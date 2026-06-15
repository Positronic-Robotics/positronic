"""Canonical PhAIL tote pick-and-place task instructions.

The single source of truth for these instruction strings, shared by the policy goal (eval configs), the recorded
`task` label (dataset configs), the operator task picker (`EvalUI`), and scoring (analysis). Dependency-free so any
of those can import it without pulling heavy deps.
"""

UNIFIED_TASK = 'Pick all the items one by one from transparent tote and place them into the large grey tote.'
TOWELS_TASK = 'Pick all the towels one by one from transparent tote and place them into the large grey tote.'
SPOONS_TASK = 'Pick all the wooden spoons one by one from transparent tote and place them into the large grey tote.'
SCISSORS_TASK = 'Pick all the scissors one by one from transparent tote and place them into the large grey tote.'
BATTERIES_TASK = 'Pick all the batteries one by one from transparent tote and place them into the large grey tote.'
