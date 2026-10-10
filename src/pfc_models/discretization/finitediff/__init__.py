"""Finite-difference discretization — boundaries are explicit, not assumed.

Planned work: semi-periodic (periodic in one axis, bounded in the other) and
fully non-periodic domains. Keeping this separate from `spectral` is what stops
boundary handling from leaking into code that is periodic by construction.
"""
