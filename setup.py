from setuptools import setup, find_packages

setup(
    name="EasyGCN", 
    version="0.1.0",
    
    author="Su-Su Zhang",
    description="A GNN extension project based on EasyGraph",
    
    packages=find_packages(),

    install_requires=[
        "Python-EasyGraph>=1.5",  
        "torch>=1.8.0",
        "numpy",
        "scipy",
        "metis",
        "fastjsonschema"
    ],
    
    python_requires=">=3.8",
)