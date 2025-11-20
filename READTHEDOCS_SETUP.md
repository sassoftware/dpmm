# Read the Docs Setup Guide for dpmm

This guide will walk you through setting up your Read the Docs website for the dpmm project.

## Prerequisites

✅ All documentation files have been created
✅ Documentation dependencies added to `pyproject.toml`
✅ `.readthedocs.yaml` configuration file created
✅ Your repository is hosted on GitHub

## Step 1: Test Documentation Locally

Before pushing to Read the Docs, test the documentation locally:

```bash
# Install documentation dependencies
poetry install --with docs

# OR using pip
pip install -r docs/requirements.txt

# Build the documentation
cd docs
make html

# Open the documentation in your browser
# The built documentation is in docs/_build/html/index.html
```

On Linux/Mac:
```bash
xdg-open _build/html/index.html  # Linux
open _build/html/index.html       # Mac
```

## Step 2: Commit and Push Changes

```bash
# Add all the new documentation files
git add docs/ .readthedocs.yaml pyproject.toml
git commit -m "Add Sphinx documentation and Read the Docs configuration"
git push origin main
```

## Step 3: Sign Up / Log In to Read the Docs

1. Go to https://readthedocs.org/
2. Click **Sign Up** or **Log In**
3. Choose **Sign in with GitHub** for easier integration
4. Authorize Read the Docs to access your GitHub repositories

## Step 4: Import Your Project

1. After logging in, click **Import a Project**
2. You should see a list of your GitHub repositories
3. Find `sassoftware/dpmm` in the list
4. Click the **+** button next to it to import

   **If you don't see your repository:**
   - Click **Import Manually**
   - Fill in the details:
     - Name: `dpmm`
     - Repository URL: `https://github.com/sassoftware/dpmm`
     - Repository type: `Git`
   - Click **Next**

## Step 5: Configure Project Settings

After importing, Read the Docs will take you to the project page:

### Basic Settings

1. Go to **Admin** → **Settings**
2. Configure the following:
   - **Name**: dpmm
   - **Description**: Differentially Private Marginal Models for Synthetic Data Generation
   - **Language**: English
   - **Programming Language**: Python
   - **Project Homepage**: https://github.com/sassoftware/dpmm
   - **Tags**: Add tags like `differential-privacy`, `synthetic-data`, `machine-learning`

3. Click **Save**

### Advanced Settings

1. Go to **Admin** → **Advanced Settings**
2. Set:
   - **Default branch**: `main`
   - **Default version**: `latest`
   - **Documentation type**: `Sphinx Html`
   - **Python configuration file**: Leave empty (it will use `.readthedocs.yaml`)

3. Click **Save**

## Step 6: Build Your Documentation

1. Go to **Builds** in the sidebar
2. Click **Build version: latest**
3. Wait for the build to complete (usually 2-5 minutes)
4. If successful, you'll see a green checkmark ✓

### If the build fails:

- Click on the failed build to see the logs
- Common issues:
  - Missing dependencies: Check `docs/requirements.txt`
  - Import errors: Make sure all imports in `conf.py` are correct
  - Syntax errors in `.rst` files

## Step 7: View Your Documentation

1. Once the build succeeds, click **View Docs** at the top right
2. Your documentation is now live at:
   - `https://dpmm.readthedocs.io/en/latest/`
   
3. Share this URL with your users!

## Step 8: Set Up Automatic Builds (Optional but Recommended)

Read the Docs automatically builds your documentation when you push to GitHub if you've connected via GitHub integration.

To verify:
1. Go to **Admin** → **Integrations**
2. You should see a GitHub webhook
3. Every push to `main` will trigger a new build

## Step 9: Configure Versions (Optional)

To build documentation for specific versions/tags:

1. Go to **Versions** in the sidebar
2. Activate the versions you want to build (e.g., stable, v0.1.9, etc.)
3. Set which version should be the default

## Step 10: Add Badge to README

Add the Read the Docs badge to your README.md:

```markdown
[![Documentation Status](https://readthedocs.org/projects/dpmm/badge/?version=latest)](https://dpmm.readthedocs.io/en/latest/?badge=latest)
```

## Maintenance

### Updating Documentation

When you update the docs:

```bash
# Edit your .rst files in docs/
# Test locally
cd docs
make html

# Commit and push
git add docs/
git commit -m "Update documentation"
git push origin main

# Read the Docs will automatically rebuild
```

### Rebuilding Manually

If automatic builds don't trigger:

1. Go to your Read the Docs project
2. Click **Builds** → **Build version: latest**
3. The documentation will rebuild

## Troubleshooting

### Build fails with "Module not found"

**Solution**: Add the missing package to `docs/requirements.txt`

### Documentation looks broken or missing content

**Solution**: 
- Check for RST syntax errors
- Verify all file references in `index.rst` exist
- Look at the build logs for warnings

### API documentation not generating

**Solution**:
- Ensure your source code has proper docstrings
- Check that `sys.path` in `conf.py` points to the right location
- Verify `autodoc` extension is enabled

### Build timeout

**Solution**:
- Simplify your documentation
- Remove heavy computations from docstrings
- Contact Read the Docs support for higher limits

## Advanced Features

### Pull Request Previews

Read the Docs can build documentation for PRs:

1. Go to **Admin** → **Advanced Settings**
2. Enable **Build pull requests for this project**
3. Each PR will get its own documentation preview

### Custom Domain

To use a custom domain like `docs.yourdomain.com`:

1. Go to **Admin** → **Domains**
2. Add your custom domain
3. Follow the DNS configuration instructions
4. Note: May require a paid plan

### Analytics

Track documentation usage:

1. Go to **Admin** → **Analytics**
2. Enable Google Analytics or use Read the Docs' built-in analytics

## Useful Links

- Read the Docs Documentation: https://docs.readthedocs.io/
- Sphinx Documentation: https://www.sphinx-doc.org/
- reStructuredText Primer: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
- Your Project Dashboard: https://readthedocs.org/projects/dpmm/

## Support

If you encounter issues:
1. Check Read the Docs documentation: https://docs.readthedocs.io/
2. Visit Read the Docs community forum: https://about.readthedocs.com/support/
3. File an issue on GitHub: https://github.com/readthedocs/readthedocs.org/issues

---

**Congratulations! Your Read the Docs site is now set up!** 🎉

Your documentation will be available at: **https://dpmm.readthedocs.io/**
