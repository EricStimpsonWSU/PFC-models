"""Array and FFT backends, one leaf module each.

Selection is explicit — a model names the backend it was written for, and a
run script may override it. There is no automatic resolution here on purpose:
silently falling back from GPU to CPU has cost more debugging time than it has
saved.
"""
