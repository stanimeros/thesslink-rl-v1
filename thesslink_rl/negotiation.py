"""Negotiation phase -- now RL-trained.

Negotiation is no longer a fixed heuristic. Instead, agents learn to
negotiate through suggest/accept actions within the episode. See
``environment.py`` for the action-space layout and agreement protocol.

The old heuristic functions (run_negotiation, _select_poi) have been
removed. The evaluation module still provides POI scoring that is used
as the negotiation observation.
"""
