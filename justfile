_default:
    @just --list --unsorted --justfile {{justfile()}} --list-prefix ····

markdown_files := "*.md"
python_files := "*.py"
yaml_files := ".github/*/*.yml"

# format Markdown, YAML and Python files
fmt:
    prettier {{markdown_files}} {{yaml_files}} --write
    isort --settings-path .github/linters/.isort.cfg {{python_files}}
    black {{python_files}}

# lint Markdown, YAML and Python files
lint:
    yamllint -c .github/linters/.yaml-lint.yml {{yaml_files}}
    prettier {{markdown_files}} {{yaml_files}} --check
    flake8 --config .github/linters/.flake8 {{python_files}}
    isort --settings-path .github/linters/.isort.cfg {{python_files}}  --check --diff
    black {{python_files}} --diff
