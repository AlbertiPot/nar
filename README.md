# Neural Architecture Ranker
This repository contains PyTorch implementation of the NAR, inculuding:

* training code and sampling code for NAR.
* detailed cell information datasets based on NAS-Bench-101 and NAS-Bench-201, and architectue encoding code.

## Cell information datasets

1. Cell information for  NAS-Bench-101

    We calcuate the FLOPs and #parameters for each node
    Dataset is available from [Google Drive](https://drive.google.com/file/d/1hM_wZzkI79tkacl3YL42ZZFAuldmGip5/view?usp=sharing), the sha256 hash is
    ```ff051bbe69e50490f8092dfc5d020675ed44e932d13619da0b6cc941f77b9c32```


2. Cell information for NAS-Bench-201
   
   Dataset is available from [Google Drive](https://drive.google.com/file/d/1MeYtWM2n-ZlUDvDyvby1lVj3hA71kZ28/view?usp=sharing), the sha256 hash is
   ```e462fa2dbff708a0d8e3f8c2bdcd5d843355d9db01cb62c5532331ad0b8ca7af```

## Train and Search