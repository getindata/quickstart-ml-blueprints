ARG BASE_IMAGE=nvidia/cuda:11.7.0-cudnn8-devel-ubuntu22.04

FROM $BASE_IMAGE

RUN apt-get update && apt-get upgrade -y

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive TZ=Europe/Warsaw apt-get install -y git make build-essential libssl-dev zlib1g-dev \
        libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
        libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

# add kedro user
ARG KEDRO_UID=1000
ARG KEDRO_GID=1000
RUN groupadd -f -g ${KEDRO_GID} kedro_group && \
    useradd -d /home/kedro -s /bin/bash -g ${KEDRO_GID} -u ${KEDRO_UID} kedro

# copy the whole project except what is in .dockerignore
ENV HOME /home/kedro
WORKDIR ${HOME}

# conda install
ENV CONDA_DIR /opt/conda
COPY install/ubuntu_install_conda.sh /install/ubuntu_install_conda.sh
RUN bash /install/ubuntu_install_conda.sh
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda update -y conda
ENV CONDA_ALWAYS_YES="true"

# poetry install
ENV POETRY_VERSION=1.2.0
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$PATH:${HOME}/.local/bin"
RUN poetry config virtualenvs.create false
COPY pyproject.toml poetry.lock ./

# conda and poetry setup
COPY environment.yml virtual-packages.yml ./
RUN conda create -n temp -c conda-forge mamba conda-lock poetry='1.*' python='3.8.12' && conda clean -afy
SHELL ["conda", "run", "-n", "temp", "/bin/bash", "-c"]
RUN conda-lock -k explicit --conda mamba
#RUN poetry add --lock torch=1.12.1 torchaudio=0.12.1 torchvision=0.13.1 conda-lock
SHELL ["/bin/bash", "-c"]
RUN conda env remove -n temp

# requirements, we can consider using conda-pack tool with multi-stage builds for smaller image size
COPY conda-linux-64.lock ./
RUN conda create --name gid_ml_framework --file conda-linux-64.lock && conda clean -afy \
    && find /opt/conda/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
    && find /opt/conda/ -follow -type f -name '*.js.map' -delete
SHELL ["conda", "run", "-n", "gid_ml_framework", "/bin/bash", "-c"]
RUN poetry lock
RUN poetry install && rm -rf ${HOME}/.cache/pypoetry

COPY . .
    
ENV PATH /opt/conda/envs/gid_ml_framework/bin:$PATH
ENV CONDA_DEFAULT_ENV gid_ml_framework

CMD ["kedro", "run", "-p", "santander_e2e"]
