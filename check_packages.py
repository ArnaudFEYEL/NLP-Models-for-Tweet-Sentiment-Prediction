import subprocess
import sys

# List of required packages
required_packages = [
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "keras"
]

def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except subprocess.CalledProcessError as e:
        print(f"Error installing {package}: {e}")

def check_and_install_packages(packages):
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"{package} is not installed. Installing...")
            install_package(package)
        else:
            print(f"{package} is already installed.")

if __name__ == "__main__":
    # Check and install required packages
    check_and_install_packages(required_packages)
    print(f"All packages needed are installed !")
