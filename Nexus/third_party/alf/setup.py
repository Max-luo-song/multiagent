# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

setup(
    name='alf',
    version='0.1.0',
    python_requires='>=3.8.0',
    install_requires=[
        'absl-py==1.4.0',
        'fasteners==0.19',
        'gin-config==0.4.0',
        'gym==0.23.1',
        'h5py==3.10.0',
        'matplotlib==3.6.1',
        'numpy==1.23.0',
        'opencv-python-headless==4.9.0.80',
        'pathos==0.2.4',
        'pillow>=8',
        'protobuf==4.25.9',
        'psutil==5.9.8',
        'pyglet==1.3.2',
        'rectangle-packer==2.0.2',
        'tensorboard==2.12.3',
        'threadpoolctl==3.2.0',
        'cnest==1.1.1',
    ],  # And any other dependencies alf needs
    ext_modules=[
        Pybind11Extension(
            'alf.environments._penv',
            sources=['alf/environments/parallel_environment.cpp'],
            extra_compile_args=[
                '-O3', '-Wall', '-std=c++17', '-fPIC', '-fvisibility=hidden'
            ])
    ],
    cmdclass={'build_ext': build_ext},
    extras_require={
        'metadrive': ['metadrive-simulator==0.2.5.1', ],
    },
    package_data={'': ['*.gin']},
    packages=find_packages(),
)
