- warns instead of throwing when validating actions

## v3.4.1

- `interactive()` mode
- fixed default arg for `context`
- fixed `say` action in `examples/quickstart.py`
- examples to take the Wit access token in argument

## v3.4.0

Unifying action parameters

### breaking

- the `say` action now takes 3 parameters: `session_id`, `context`, `msg`
- the `error` action now takes 3 parameters: `session_id`, `context`, `e`

## v3.3.0

Updating action parameters

### breaking

- the `merge` action now takes 4 parameters: `session_id`, `context`, `entities`, `msg`
- the `error` action now takes `context` as second parameter
- custom actions now take 2 parameters: `session_id`, `context`

## v3.2

- Fixed request keyword arguments issue
- Better error messages

## v3.1

- Added `examples/template.py`
- Fixed missing type
- Updated `examples/weather.py` to `examples/quickstart.py` to reflect the docs

## v3.0

Bot Engine integration

### breaking

- the `message` API is wrapped around a `Wit` class, and doesn't take the token as first parameter

## v2.0

Rewrite in pure Python

### breaking

- audio recording and streaming have been removed because:
  - many people only needed access to the HTTP API, and audio recording did not make sense for server-side use cases
  - dependent on platform, choice best left to developers
  - forced us to maintain native bindings as opposed to a pure Pythonic library
- we renamed the functions to match the HTTP API more closely
  - `.text_query(string, access_token)` becomes `.message(access_token, string)`
- all functions now return a Python dict instead of a JSON string
