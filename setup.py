from setuptools import setup, find_packages

setup(
    name="amharic-ecommerce-extractor",
    version="0.1.0",
    description="Amharic E-commerce Data Extractor for EthioMart",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "datasets>=2.14.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "shap>=0.42.0",
        "lime>=0.2.0.1",
        "python-dotenv>=1.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
