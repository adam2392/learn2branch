# SCIP solver

Set-up a desired installation path for SCIP / SoPlex (e.g., `/opt/scip`):
```
export SCIPOPTDIR='/opt/scip'
```

c

SoPlex 4.0.1 (free for academic uses)

https://soplex.zib.de/download.php?fname=soplex-4.0.1.tgz

```
tar -xzf soplex-4.0.1.tgz
cd soplex-4.0.1/
mkdir build
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$SCIPOPTDIR
make -C ./build -j 4
make -C ./build install
cd ..
```

# SCIP

SCIP 6.0.1 (free for academic uses)

https://scip.zib.de/download.php?fname=scip-6.0.1.tgz

```
tar -xzf scip-6.0.1.tgz
cd scip-6.0.1/
```

Apply patch file in `learn2branch/scip_patch/`

```
patch -p1 < ../learn2branch/scip_patch/vanillafullstrong.patch
```

```
mkdir build
cmake -S . -B build -DSOPLEX_DIR=$SCIPOPTDIR -DCMAKE_INSTALL_PREFIX=$SCIPOPTDIR
make -C ./build -j 4
make -C ./build install
cd ..
```

For reference, original installation instructions [here](http://scip.zib.de/doc/html/CMAKE.php).

# Python dependencies

Recommended setup: conda + python 3

https://docs.conda.io/en/latest/miniconda.html

## Cython

Required to compile PySCIPOpt and PySVMRank
```
conda install cython
```

## PySCIPOpt

SCIP's python interface (modified version)

```
pip install git+https://github.com/ds4dm/PySCIPOpt.git@ml-branching
```

## ExtraTrees
```
conda install scikit-learn=0.20.2  # ExtraTrees
```

## LambdaMART
```
pip install git+https://github.com/jma127/pyltr@78fa0ebfef67d6594b8415aa5c6136e30a5e3395  # LambdaMART
```

## SVMrank
```
git clone https://github.com/ds4dm/PySVMRank.git
cd PySVMRank
wget http://download.joachims.org/svm_rank/current/svm_rank.tar.gz  # get SVMrank original source code
mkdir src/c
tar -xzf svm_rank.tar.gz -C src/c
pip install .
```

## Tensorflow
```
conda install tensorflow-gpu=1.12.0
```

# General Install (MARCC)
Note MARCC can be buggy due to lack of sudo. So we put SCIPOPTDIR in a local directory attached to a user. We also need the cmake module
because it is not by default avail. for all users. Note you need to make sure cmake is at least version (3.15?); 
needs to be upgraded usually or else some of the settings do not work as expected.
   
    # load modules needed
    ml cmake
    
    # create scip directory in code to build
    cd ~/code/
    mkdir scip/
    
    # pull repo down
    git clone https://github.com/adam2392/learn2branch.git
    cd learn2branch
    
    # copy scip and soplex tar to outside
    cp ./scip-6.0.1.tgz ../
    cp ./soplex-4.0.1.tgz ../
    
Now follow Soplex and SCIP installation instructions above.

    # remove the untarred dirs
    rm -rf ./scip-6.0.1/
    rm -rf ./soplex-4.0.1/
    rm ./scip-6.0.1.tgz
    rm ./soplex-4.0.1.tgz
    
# General Install (Conda)

    conda create -n learn2branch python=3.5
    conda install cython numpy scikit-learn scipy tqdm natsort joblib
    # pip install tensorflow-gpu==1.12.0
    # pip install cython numpy scikit-learn scipy tqdm natsort joblib
    # LambdaMART
    pip install git+https://github.com/jma127/pyltr@78fa0ebfef67d6594b8415aa5c6136e30a5e3395  
    pip install git+https://github.com/ds4dm/PySCIPOpt.git@ml-branching
    