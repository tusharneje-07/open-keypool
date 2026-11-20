# Contributing

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev,env]"
```

## Tests

Run `pytest`. The CI matrix tests Python 3.9 through 3.13.

## Adding a provider preset

Verify the provider's official status codes, error body fields, and retry
headers first. Add only documented signals to the data-driven rules in
`src/open_keypool/core.py`, then add sync and async tests. Unknown provider
names must continue to use the generic rules.
