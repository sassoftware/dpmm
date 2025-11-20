# Quick Reference: Documentation Commands

## Local Development

### Build Documentation
```bash
cd docs
make html
```

### Clean Build
```bash
cd docs
make clean
make html
```

### View Documentation
```bash
# Linux
xdg-open _build/html/index.html

# Mac
open _build/html/index.html

# Windows
start _build/html/index.html
```

### Watch for Changes (requires sphinx-autobuild)
```bash
pip install sphinx-autobuild
cd docs
sphinx-autobuild . _build/html
# Opens browser automatically at http://localhost:8000
```

## Common Make Commands

```bash
make html      # Build HTML documentation
make clean     # Remove built documentation
make latexpdf  # Build PDF (requires LaTeX)
make epub      # Build EPUB
make linkcheck # Check for broken links
make doctest   # Run doctests
make help      # Show all available commands
```

## File Structure

```
docs/
├── conf.py              # Sphinx configuration
├── index.rst            # Main documentation page
├── installation.rst     # Installation guide
├── quickstart.rst       # Quick start guide
├── user_guide.rst       # Comprehensive user guide
├── api.rst              # API reference
├── examples.rst         # Examples and use cases
├── contributing.rst     # Contributing guide
├── license.rst          # License information
├── requirements.txt     # Documentation dependencies
├── Makefile            # Build commands
├── _static/            # Static files (CSS, images)
├── _templates/         # Custom templates
└── _build/             # Built documentation (gitignored)
```

## Adding New Pages

1. Create a new `.rst` file in `docs/`
2. Add it to the `toctree` in `index.rst`:

```rst
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   your_new_page  # Add here
   api
```

## RST Quick Reference

### Headers
```rst
Page Title
==========

Section
-------

Subsection
~~~~~~~~~~

Subsubsection
^^^^^^^^^^^^^
```

### Code Blocks
```rst
.. code-block:: python

   import dpmm
   model = dpmm.pipelines.MSTPipeline()
```

### Links
```rst
`Link text <https://example.com>`_
:doc:`Other page <installation>`
```

### Lists
```rst
- Item 1
- Item 2
  - Nested item

1. Numbered item
2. Another item
```

### Admonitions
```rst
.. note::
   This is a note

.. warning::
   This is a warning

.. seealso::
   Related information
```

## Debugging

### Check for errors
```bash
cd docs
make clean
make html 2>&1 | tee build.log
```

### Common Issues

**Import errors:**
- Check `sys.path` in `conf.py`
- Ensure source code is accessible

**Missing references:**
- Check file names match exactly
- Use `:doc:` for internal links

**RST syntax errors:**
- Use proper indentation (3 spaces for nested content)
- Check for missing blank lines

## Read the Docs

### Manual Build
1. Go to https://readthedocs.org/projects/dpmm/
2. Click "Builds" → "Build version: latest"

### View Logs
1. Click on any build
2. View raw logs or build output

### Update Settings
Admin → Settings → Make changes → Save
