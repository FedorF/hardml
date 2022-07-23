FROM ubuntu:18.04 

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \ 
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*

RUN curl -L -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \     
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install conda-build && \
     /opt/conda/bin/conda install numpy==1.19.2 pyyaml==5.4.1 scipy==1.6.1 ipython flask==1.1.2 mkl pandas==1.2.3 && \
     /opt/conda/bin/conda clean -ya 
ENV PATH /opt/conda/bin:$PATH
RUN conda install scikit-learn==0.23.2 catboost==0.23 -c conda-forge && /opt/conda/bin/conda clean -ya

WORKDIR /workspace
RUN chmod -R a+w /workspace
