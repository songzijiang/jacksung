python setup.py sdist bdist_wheel
twine upload dist/*
pip install jacksung --upgrade