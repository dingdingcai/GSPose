pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 -f https://download.pytorch.org/whl/torch_stable.html
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install xformers==0.0.22

cd submodules/Connected_components_PyTorch
python setup.py install

cd ../diff-gaussian-rasterization
python setup.py install

cd ../simple-knn
python setup.py install

cd ../../model/curope
python setup.py install
cd ../..
