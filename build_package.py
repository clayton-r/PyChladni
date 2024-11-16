import os
import shutil

# Clean previous builds
if os.path.exists("build"):
    shutil.rmtree("build")
if os.path.exists("dist"):
    shutil.rmtree("dist")
if os.path.exists("pychladni.egg-info"):
    shutil.rmtree("pychladni.egg-info")

# Build distributions
os.system("python -m build")

print("\nBuilt package. To upload to PyPI, use:")
print("twine upload dist/*")