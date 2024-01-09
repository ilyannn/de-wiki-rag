_default:
    just --list

fmt:
    isort --settings-path .github/linters/.isort.cfg .
    black .
    prettier --write *.md

lint:
    yamllint -c .github/linters/.yaml-lint.yml .
    flake8 --config .github/linters/.flake8 *.py
    isort --settings-path .github/linters/.isort.cfg .  --check --diff
    black . --diff
    prettier *.md 
