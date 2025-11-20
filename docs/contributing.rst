Contributing
============

We welcome contributions to dpmm! This guide will help you get started.

Getting Started
---------------

1. Fork the repository on GitHub
2. Clone your fork locally:

   .. code-block:: bash

      git clone git@github.com:YOUR_USERNAME/dpmm.git
      cd dpmm

3. Set up the development environment:

   .. code-block:: bash

      poetry install --with dev

4. Create a branch for your changes:

   .. code-block:: bash

      git checkout -b feature/your-feature-name


Development Guidelines
----------------------

Code Style
~~~~~~~~~~

We follow PEP 8 style guidelines. The project uses:

- **black** for code formatting
- **flake8** for linting
- **ruff** for additional linting

Run formatters before committing:

.. code-block:: bash

   poetry run black src/
   poetry run ruff check src/


Testing
~~~~~~~

All new features should include tests. Run the test suite:

.. code-block:: bash

   pytest tests/

Run tests with coverage:

.. code-block:: bash

   pytest tests/ --cov=src/dpmm --cov-report=html

Run tests in parallel:

.. code-block:: bash

   pytest tests/ -n auto


Documentation
~~~~~~~~~~~~~

Update documentation for any new features or changes:

1. Add docstrings to new functions/classes (Google or NumPy style)
2. Update relevant .rst files in the docs/ directory
3. Add examples if applicable

Build documentation locally:

.. code-block:: bash

   cd docs
   make html
   # Open _build/html/index.html in your browser


Pull Request Process
--------------------

1. **Update your branch** with the latest main:

   .. code-block:: bash

      git checkout main
      git pull upstream main
      git checkout your-branch
      git rebase main

2. **Run tests and linting**:

   .. code-block:: bash

      pytest tests/
      poetry run flake8 src/
      poetry run black src/ --check

3. **Commit your changes** with descriptive messages:

   .. code-block:: bash

      git add .
      git commit -m "Add feature: brief description"

4. **Push to your fork**:

   .. code-block:: bash

      git push origin your-branch

5. **Create a Pull Request** on GitHub:

   - Provide a clear description of the changes
   - Reference any related issues
   - Ensure all CI checks pass


Types of Contributions
----------------------

Bug Reports
~~~~~~~~~~~

When reporting bugs, please include:

- Python version
- dpmm version
- Operating system
- Minimal code to reproduce the issue
- Error messages and tracebacks


Feature Requests
~~~~~~~~~~~~~~~~

For new features:

- Explain the use case
- Describe the proposed API
- Discuss alternatives you've considered


Code Contributions
~~~~~~~~~~~~~~~~~~

We welcome:

- Bug fixes
- New features
- Performance improvements
- Documentation improvements
- Test coverage improvements


Code Review Process
-------------------

All submissions require review. We use GitHub pull requests for this purpose.

Reviewers will check:

- Code quality and style
- Test coverage
- Documentation completeness
- Compatibility with existing code


Contributor Agreement
---------------------

By contributing to this project, you agree to the terms in the
`Contributor Agreement <https://github.com/sassoftware/dpmm/tree/main/ContributorAgreement.txt>`_.


Community
---------

- GitHub Issues: `<https://github.com/sassoftware/dpmm/issues>`_
- GitHub Discussions: For questions and general discussion


Getting Help
------------

If you need help:

1. Check the documentation
2. Search existing GitHub issues
3. Create a new issue with the "question" label


Thank You!
----------

Thank you for contributing to dpmm! Your efforts help make
differential privacy more accessible to everyone.
