from setuptools import setup, find_packages

setup(
    name='xarm-gym',
    version='0.1',
    packages=find_packages(),
    description='gym based environment for real robot',
    author='Matt Hyatt',
    author_email='mhyatt000@gmail.com',
    url='https://github.com/mhyatt000/xarm-gym',
    install_requires=[
        "xarm @ git+https://github.com/xArm-Developer/xArm-Python-SDK.git"
        "gello @ git+https://github.com/wuphilipp/gello_software.git",
        "gymnasium",
        "numpy",
        "opencv-python",
        "pyrealsense2",
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
