"""Multi-file port of the mini-swe-agent control loop for the tau subnet.

The package keeps the same building blocks as the upstream project
(https://github.com/SWE-agent/mini-swe-agent): a model wrapper, a bash
execution environment, prompt templates, and a step loop. Everything is
standard-library only and all inference goes through the validator-managed
OpenAI-compatible proxy passed into agent.solve().
"""
