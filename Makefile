lint:
\truff .

type:
\tmypy src

test:
\tpytest

all: lint type test
