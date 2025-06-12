python setup.py sdist bdist_wheel
twine upload dist/*
git push
pip install jacksung --upgrade
